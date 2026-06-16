#!/usr/bin/env python3
"""Download GEO Perturb-seq datasets in a format-aware way.

This script lives under ``perturbseq/analysis/`` and never modifies anything
under ``perturbseq/model/``.

The seven target series do NOT share a single layout. Each is handled
according to how its authors deposited the data on GEO (discovered by
inspecting the series ``suppl/`` listings):

  GSE142078  raw_tar_legacy    GSE142078_RAW.tar -> per-GSM legacy CellRanger v2
                               triples (barcodes/genes/matrix) + a separate
                               *_Cell_Guide_Lookup.csv.gz (gene expr only;
                               guides live in the lookup CSV).
  GSE208240  series_tar        one big *_filtered.tar.gz; GEX and gRNA are
                               deposited as SEPARATE matrices (GSM ..,gex /
                               ..,guide) that 02.prepare_h5ad.py merges.
  GSE252965  atac_incompatible ATAC-seq bigWig only -> NOT a Perturb-seq
                               gene+guide dataset. Skipped with a warning.
  GSE272457  series_triples    several series-level 10X triples, one per
                               sample prefix; matrix already multi-feature
                               (Gene Expression + CRISPR Guide Capture).
  GSE278572  series_triple     one series-level multi-feature 10X triple
                               (+ protospacer_calls).
  GSE280506  series_h5         filtered_feature_bc_matrix.h5 (multi-feature)
                               + cell_identities.csv + guide-library xlsx.
  GSE311503  series_triples    two series-level multi-feature triples (D1, D2).

What "download the core data" means here: fetch whatever lets
``perturbseq/model/prepare_perturb_h5ad.py`` ultimately run, i.e. a 10X count
matrix (Gene Expression + CRISPR Guide Capture) plus the guide assignment
information. Reshaping heterogeneous inputs into that canonical form is the job
of ``02.prepare_h5ad.py``.

Usage:
    python 01.download_geo.py --series GSE311503 --outdir perturbseq/analysis/00.data
    python 01.download_geo.py --series GSE142078,GSE278572,GSE311503 \
        --outdir perturbseq/analysis/00.data --extract
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

GEO_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"

# ---------------------------------------------------------------------------
# Per-series configuration (the "kind" drives which files are fetched).
# ---------------------------------------------------------------------------
SERIES_CONFIG: dict[str, dict] = {
    "GSE142078": {
        "kind": "raw_tar_legacy",
        "compatible": True,
        "note": "Legacy CellRanger v2 triples + separate Cell_Guide_Lookup.csv.",
    },
    "GSE208240": {
        "kind": "series_tar",
        "compatible": True,
        "note": "Big *_filtered.tar.gz with SEPARATE GEX and guide matrices.",
    },
    "GSE252965": {
        "kind": "atac_incompatible",
        "compatible": False,
        "note": "ATAC-seq bigWig only; not a Perturb-seq gene+guide matrix.",
    },
    "GSE272457": {
        "kind": "series_triples",
        "compatible": True,
        "note": "Multiple multi-feature 10X triples (one per sample prefix).",
    },
    "GSE278572": {
        "kind": "series_triple",
        "compatible": True,
        "note": "One multi-feature 10X triple (+ protospacer_calls).",
    },
    "GSE280506": {
        "kind": "series_h5",
        "compatible": True,
        "note": "filtered_feature_bc_matrix.h5 + cell_identities.csv.",
    },
    "GSE311503": {
        "kind": "series_triples",
        "compatible": True,
        "note": "Two multi-feature 10X triples (D1, D2).",
    },
}

# File-name patterns used to pick the relevant files out of a suppl listing.
TRIPLE_SUFFIXES = ("_barcodes.tsv.gz", "_features.tsv.gz", "_matrix.mtx.gz")
BARE_TRIPLE = ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")
# Auxiliary guide/identity files that are cheap and useful to keep.
AUX_PATTERNS = (
    "protospacer_calls",
    "cell_identities",
    "cell_guide_lookup",
    "guide",
    "identities",
    "reference",
)


# ---------------------------------------------------------------------------
# GEO suppl directory listing
# ---------------------------------------------------------------------------
class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.hrefs.append(value)


def series_suppl_url(series: str) -> str:
    """Build the GEO FTP suppl/ URL for a series (e.g. GSE142nnn/GSE142078)."""
    digits = series[3:]
    bucket = f"GSE{digits[:-3]}nnn"
    return f"{GEO_FTP_BASE}/{bucket}/{series}/suppl/"


def list_suppl_files(series: str) -> list[str]:
    """Return the supplementary file names available for a series."""
    url = series_suppl_url(series)
    with urllib.request.urlopen(url, timeout=120) as resp:
        html = resp.read().decode("utf-8", "replace")
    parser = _HrefParser()
    parser.feed(html)
    files: list[str] = []
    for href in parser.hrefs:
        name = Path(urlsplit(href).path).name
        if not name:
            continue
        # Keep only this series' files (GSE-prefixed) or bare 10X triple names;
        # drop navigation/policy links and the directory's own entries.
        if not (name.startswith(series) or name in BARE_TRIPLE):
            continue
        if name.lower().endswith((".html", ".htm")):
            continue
        if name not in files:
            files.append(name)
    return files


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def download_file(url: str, target: Path, *, retries: int = 3) -> bool:
    """Download ``url`` to ``target``; skip if a complete copy already exists."""
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(url)
    if target.exists():
        local = target.stat().st_size
        if expected is None or local == expected:
            print(f"  [skip] {target.name} already present ({local:,} bytes)")
            return True
        print(f"  [redo] {target.name} size {local:,} != remote {expected:,}; re-downloading")

    size_str = f"{expected:,} bytes" if expected else "unknown size"
    for attempt in range(1, retries + 1):
        try:
            print(f"  [get ] {target.name} ({size_str})  attempt {attempt}/{retries}")
            tmp = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)
            return True
        except Exception as exc:  # noqa: BLE001 - report and retry
            print(f"        failed: {exc}")
    print(f"  [FAIL] could not download {url}")
    return False


def select_files(kind: str, files: list[str]) -> list[str]:
    """Pick the file names to download for a given series ``kind``."""
    low = {f: f.lower() for f in files}

    def is_triple(f: str) -> bool:
        return f.endswith(TRIPLE_SUFFIXES) or f in BARE_TRIPLE

    def is_aux(f: str) -> bool:
        return any(p in low[f] for p in AUX_PATTERNS) and f.lower().endswith(
            (".csv.gz", ".tsv.gz", ".txt.gz", ".csv", ".xlsx")
        )

    if kind in ("series_triple", "series_triples"):
        chosen = [f for f in files if is_triple(f)]
        chosen += [f for f in files if is_aux(f) and f not in chosen]
        return chosen
    if kind == "series_h5":
        chosen = [f for f in files if f.lower().endswith(".h5")]
        chosen += [f for f in files if is_aux(f) and f not in chosen]
        # also keep the bare triple if present (GEX-only fallback)
        chosen += [f for f in files if is_triple(f) and f not in chosen]
        return chosen
    if kind in ("series_tar", "raw_tar_legacy"):
        return [f for f in files if f.lower().endswith((".tar", ".tar.gz", ".tgz"))]
    return []


def extract_archives(series_dir: Path) -> None:
    """Extract any *.tar / *.tar.gz found directly under ``series_dir``."""
    for archive in sorted(series_dir.glob("*.tar*")):
        if not tarfile.is_tarfile(archive):
            continue
        dest = series_dir / (archive.name.split(".tar")[0] + "_extracted")
        if dest.exists() and any(dest.iterdir()):
            print(f"  [skip] already extracted: {dest.name}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  [tar ] extracting {archive.name} -> {dest.name}/")
        with tarfile.open(archive) as tar:
            _safe_extractall(tar, dest)


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract guarding against path traversal (CVE-2007-4559)."""
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest)):
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    tar.extractall(dest)


# ---------------------------------------------------------------------------
# Per-series driver
# ---------------------------------------------------------------------------
def download_series(series: str, outdir: Path, extract: bool) -> Path | None:
    cfg = SERIES_CONFIG[series]
    kind = cfg["kind"]
    series_dir = outdir / series
    print(f"\n=== {series}  [{kind}] ===")
    print(f"    {cfg['note']}")

    if kind == "atac_incompatible":
        print(
            f"  [WARN] {series} is {cfg['note']} It cannot be turned into a\n"
            f"         Perturb-seq h5ad by prepare_perturb_h5ad.py and is SKIPPED.\n"
            f"         (Remove it from --series, or supply the matching gene+guide\n"
            f"          matrices separately if you have them.)"
        )
        return None

    series_dir.mkdir(parents=True, exist_ok=True)
    base = series_suppl_url(series)

    try:
        available = list_suppl_files(series)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] could not list suppl/ for {series}: {exc}")
        return None

    wanted = select_files(kind, available)
    if not wanted:
        print(f"  [WARN] no matching files found in suppl/ for kind '{kind}'.")
        print(f"         available: {available}")
        return None

    ok = True
    for name in wanted:
        ok &= download_file(base + name, series_dir / name)

    if extract and kind in ("series_tar", "raw_tar_legacy"):
        extract_archives(series_dir)

    print(f"  done: {series_dir}  ({'OK' if ok else 'with errors'})")
    return series_dir


def parse_series(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GEO Perturb-seq datasets (format-aware).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--series",
        type=str,
        required=True,
        help="Comma-separated GEO series IDs (e.g. GSE311503,GSE278572). "
        "Use 'all' for every configured series.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output root; each series goes under <outdir>/<GSE>/.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded *.tar/*.tar.gz archives (raw_tar / series_tar).",
    )
    args = parser.parse_args()

    if args.series.strip().lower() == "all":
        series_list = sorted(SERIES_CONFIG)
    else:
        series_list = parse_series(args.series)

    unknown = [s for s in series_list if s not in SERIES_CONFIG]
    if unknown:
        print(
            f"WARNING: unconfigured series will be skipped: {', '.join(unknown)}\n"
            f"         configured: {', '.join(sorted(SERIES_CONFIG))}"
        )

    results: dict[str, str] = {}
    for series in series_list:
        if series not in SERIES_CONFIG:
            continue
        path = download_series(series, args.outdir, args.extract)
        results[series] = str(path) if path else "skipped/failed"

    print("\n=== summary ===")
    for series in series_list:
        if series in SERIES_CONFIG:
            print(f"  {series}: {results.get(series, 'n/a')}")


if __name__ == "__main__":
    main()
