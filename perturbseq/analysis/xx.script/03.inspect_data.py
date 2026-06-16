#!/usr/bin/env python3
"""Inspect the state of downloaded (raw) and normalized (processed) Perturb-seq
datasets and write a report.

For every dataset/sample it records:
  * stage (raw / processed) and the files present
  * number of cells (barcodes)
  * number of features, split into Gene Expression and CRISPR Guide Capture
  * number of guide RNAs, distinct target genes, and non-targeting (NT) guides
    (from guide_map.csv when available)
  * a guide-dominance check (``--deep``): the single guide carrying the largest
    share of guide UMIs, its UMI fraction and how ubiquitous it is across cells
    -- this is what surfaces pathological guides like FOXO4.4 in GSE311503 D1.

Outputs (under <root>/logs/):
  * data_inspection.tsv          -- one row per dataset/sample (machine-readable)
  * data_inspection_report.md    -- human-readable summary
Anomalies found are also appended to <root>/logs/anomalies.md.

Usage:
    python 03.inspect_data.py --root perturbseq/analysis/00.data --deep
    python 03.inspect_data.py --root perturbseq/analysis/00.data --series GSE311503
"""
from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import geo_utils

GUIDE_FT = "CRISPR Guide Capture"
GENE_FT = "Gene Expression"

# Guide-dominance thresholds for flagging an anomaly.
DOM_UMI_FRAC = 0.50      # one guide is >50% of all guide UMIs
DOM_UBIQUITY = 0.90      # ...and detected in >90% of cells
DEFAULT_MAX_DEEP_BYTES = 600_000_000  # skip the deep load above this matrix size


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------
def _read_lines_gz(path: Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def _count_lines_gz(path: Path) -> int:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return sum(1 for ln in fh if ln.strip())


def _feature_type_counts(features_path: Path) -> tuple[int, int, int]:
    """Return (n_features, n_gene_expr, n_guides) from a features/genes file."""
    rows = _read_lines_gz(features_path)
    n_total = len(rows)
    n_guide = n_gex = 0
    has_type = False
    for r in rows:
        cols = r.split("\t")
        if len(cols) >= 3:
            has_type = True
            if cols[2] == GUIDE_FT:
                n_guide += 1
            elif cols[2] == GENE_FT:
                n_gex += 1
    if not has_type:
        # legacy 2-column genes.tsv -> all gene expression, guides live elsewhere
        n_gex = n_total
    return n_total, n_gex, n_guide


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------
def _find_named(series_dir: Path, key: str, *, suffixes: tuple[str, ...] = ()) -> Path | None:
    for p in sorted(series_dir.rglob("*")):
        if not p.is_file() or key not in p.name.lower():
            continue
        if suffixes and not p.name.lower().endswith(suffixes):
            continue
        return p
    return None


def discover_samples(series_dir: Path) -> list[dict]:
    """Return a list of {sample, kind, parts/h5/...} entries under a series dir."""
    entries: list[dict] = []
    for sample, parts in geo_utils.find_triples(series_dir).items():
        entries.append({"sample": sample, "kind": "triple", "parts": parts})
    for h5 in sorted(series_dir.rglob("*.h5")):
        entries.append({"sample": h5.stem, "kind": "h5", "h5": h5})
    # GSE236057-style raw layout: non-standard-named GEX matrix + a metadata CSV
    # that embeds a wide boolean guide-by-cell block (not caught by find_triples).
    counts = _find_named(series_dir, "counts.mtx")
    if counts is not None:
        entries.append({
            "sample": re.split(r"_[Cc]ounts", counts.name)[0],
            "kind": "named_matrix",
            "barcodes": _find_named(series_dir, "barcodes", suffixes=(".tsv.gz", ".tsv")),
            "genenames": _find_named(series_dir, "genenames"),
            "metadata": _find_named(series_dir, "metadata", suffixes=(".csv.gz", ".csv")),
        })
    return entries


def _target_from_guide(col: str) -> str:
    """Lightweight GSE236057 guide->target (mirrors 02's enh_pos_neg parsing)."""
    if re.match(r"(?i)^neg", col):
        return "NT"
    if col.startswith("Pos_"):
        return col[len("Pos_"):]
    return re.split(r"_g\d+", col)[0]


# ---------------------------------------------------------------------------
# Per-sample inspection
# ---------------------------------------------------------------------------
def _guide_dominance_from_matrix(
    guide_cells_x_guides: sparse.spmatrix, guide_names: list[str]
) -> dict:
    g = sparse.csr_matrix(guide_cells_x_guides)
    total = float(g.sum())
    if total <= 0:
        return {}
    per_guide = np.asarray(g.sum(axis=0)).ravel()
    ubiquity = np.asarray((g > 0).sum(axis=0)).ravel() / max(1, g.shape[0])
    top = int(np.argmax(per_guide))
    return {
        "top_guide": guide_names[top],
        "top_guide_umi_frac": round(per_guide[top] / total, 4),
        "top_guide_ubiquity": round(float(ubiquity[top]), 4),
    }


def inspect_sample(
    series: str, stage: str, entry: dict, *, deep: bool, max_deep_bytes: int, root: Path
) -> dict:
    sample = entry["sample"]
    rec = {
        "series": series, "stage": stage, "sample": sample,
        "n_cells": None, "n_features": None, "n_gene_expr": None, "n_guides": None,
        "n_targets": None, "n_nt": None,
        "top_guide": "", "top_guide_umi_frac": None, "top_guide_ubiquity": None,
        "notes": "",
    }
    notes: list[str] = []

    if entry["kind"] == "triple":
        parts = entry["parts"]
        rec["n_cells"] = _count_lines_gz(parts["barcodes"])
        if "features" in parts:
            n_f, n_gex, n_guide = _feature_type_counts(parts["features"])
            rec["n_features"], rec["n_gene_expr"], rec["n_guides"] = n_f, n_gex, n_guide
            if n_guide == 0:
                notes.append("no CRISPR features in matrix (guides likely in a separate file)")
        # guide_map.csv (present in processed dirs)
        gm = parts["barcodes"].parent / "guide_map.csv"
        if gm.exists():
            df = pd.read_csv(gm)
            rec["n_guides"] = rec["n_guides"] or len(df)
            rec["n_targets"] = int(df["target_gene"].nunique())
            rec["n_nt"] = int((df["target_gene"] == "NT").sum())
        # deep guide-dominance check
        if deep and "features" in parts and (rec["n_guides"] or 0) > 0:
            size = parts["matrix"].stat().st_size
            if size > max_deep_bytes:
                notes.append(f"deep skipped (matrix {size/1e6:.0f}MB > {max_deep_bytes/1e6:.0f}MB)")
            else:
                adata = geo_utils.load_triple_adata(parts)
                ft = adata.var["feature_types"].astype(str)
                gmask = ft.eq(GUIDE_FT).to_numpy()
                if gmask.any():
                    dom = _guide_dominance_from_matrix(
                        adata[:, gmask].X, adata.var_names[gmask].astype(str).tolist()
                    )
                    rec.update(dom)

    elif entry["kind"] == "named_matrix":
        if entry.get("barcodes"):
            rec["n_cells"] = _count_lines_gz(entry["barcodes"])
        if entry.get("genenames"):
            # GEX-only matrix; guides live in the metadata CSV, not here.
            n_g = _count_lines_gz(entry["genenames"])
            rec["n_features"], rec["n_gene_expr"] = n_g, n_g
        meta = entry.get("metadata")
        if meta is not None:
            hdr = list(pd.read_csv(meta, nrows=0).columns)
            gcols = [c for c in hdr if re.match(r"(?i)^(enh|pos|neg)", str(c))]
            if gcols:
                rec["n_guides"] = len(gcols)
                targets = {_target_from_guide(c) for c in gcols}
                rec["n_targets"] = len(targets)
                rec["n_nt"] = sum(1 for c in gcols if re.match(r"(?i)^neg", str(c)))
                notes.append("guides embedded as a wide boolean matrix in the "
                             "metadata CSV (synthesized into a CRISPR block by 02)")
        else:
            notes.append("no metadata CSV found; guide block cannot be located")

    elif entry["kind"] == "h5":
        adata = sc.read_10x_h5(entry["h5"], gex_only=False)
        adata.var_names_make_unique()
        rec["n_cells"] = int(adata.n_obs)
        ft = adata.var.get("feature_types")
        if ft is not None:
            ft = ft.astype(str)
            rec["n_features"] = int(adata.n_vars)
            rec["n_gene_expr"] = int(ft.eq(GENE_FT).sum())
            rec["n_guides"] = int(ft.eq(GUIDE_FT).sum())
            if deep and rec["n_guides"]:
                gmask = ft.eq(GUIDE_FT).to_numpy()
                dom = _guide_dominance_from_matrix(
                    adata[:, gmask].X, adata.var_names[gmask].astype(str).tolist()
                )
                rec.update(dom)
        else:
            notes.append("h5 has no feature_types column")

    # anomaly flags
    frac = rec["top_guide_umi_frac"]
    ubiq = rec["top_guide_ubiquity"]
    if frac is not None and ubiq is not None and frac >= DOM_UMI_FRAC and ubiq >= DOM_UBIQUITY:
        notes.append(
            f"DOMINANT GUIDE {rec['top_guide']} = {frac:.0%} of guide UMIs, "
            f"in {ubiq:.0%} of cells"
        )
        geo_utils.append_anomaly(
            root, series,
            f"Pathologically dominant guide ({rec['top_guide']})",
            observation=(
                f"[{stage}/{sample}] guide '{rec['top_guide']}' carries {frac:.0%} of "
                f"all guide UMIs and is present in {ubiq:.0%} of cells -- it swamps the "
                "dominant-guide call, so almost every cell is assigned to its target."
            ),
            action=(
                "Not a pipeline bug (faithful to the GEO matrix). Before guide calling, "
                "consider dropping this guide or this sample, or use a different sample "
                "(e.g. GSE311503 D2) for per-target analysis."
            ),
        )
    if rec["n_nt"] == 0 and rec["n_guides"]:
        notes.append("guide_map has 0 NT guides")

    rec["notes"] = "; ".join(notes)
    return rec


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def write_reports(records: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    tsv = out_dir / "data_inspection.tsv"
    df.to_csv(tsv, sep="\t", index=False)

    md = out_dir / "data_inspection_report.md"
    lines = ["# Data inspection report", ""]
    if df.empty:
        lines.append("_No datasets found. Run 01.download_geo.py / 02.prepare_h5ad.py first._")
    else:
        for series in sorted(df["series"].unique()):
            lines.append(f"## {series}")
            sub = df[df["series"] == series]
            for _, r in sub.iterrows():
                dom = ""
                if r["top_guide"]:
                    dom = (f"  \n    top guide: `{r['top_guide']}` "
                           f"({_fmt_pct(r['top_guide_umi_frac'])} of guide UMIs, "
                           f"{_fmt_pct(r['top_guide_ubiquity'])} of cells)")
                lines.append(
                    f"- **[{r['stage']}] {r['sample']}** — "
                    f"cells={_fmt(r['n_cells'])}, features={_fmt(r['n_features'])} "
                    f"(GEX={_fmt(r['n_gene_expr'])}, guides={_fmt(r['n_guides'])}), "
                    f"targets={_fmt(r['n_targets'])}, NT={_fmt(r['n_nt'])}{dom}"
                )
                if r["notes"]:
                    lines.append(f"    - notes: {r['notes']}")
            lines.append("")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tsv, md


def _fmt(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{int(v):,}"


def _fmt_pct(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.0%}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect raw/processed Perturb-seq datasets; write a report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="Data root containing raw/ and/or processed/ (01 --outdir / 02 --out-root).")
    parser.add_argument("--series", default=None,
                        help="Comma-separated GEO IDs to inspect (default: all found).")
    parser.add_argument("--which", choices=["raw", "processed", "both"], default="both",
                        help="Which stage(s) to inspect.")
    parser.add_argument("--deep", action="store_true",
                        help="Load matrices to compute guide-dominance stats (slower).")
    parser.add_argument("--max-deep-bytes", type=int, default=DEFAULT_MAX_DEEP_BYTES,
                        help="Skip the deep load for matrix.mtx.gz larger than this.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Report output dir (default: <root>/logs).")
    args = parser.parse_args()

    log = geo_utils.get_logger(args.root, "inspect")
    out_dir = args.out or geo_utils.logs_root(args.root)
    want = {s.strip() for s in args.series.split(",")} if args.series else None

    stage_roots = []
    if args.which in ("raw", "both"):
        stage_roots.append(("raw", geo_utils.raw_root(args.root)))
    if args.which in ("processed", "both"):
        stage_roots.append(("processed", geo_utils.processed_root(args.root)))

    records: list[dict] = []
    for stage, stage_root in stage_roots:
        if not stage_root.exists():
            log.info("(%s root not found: %s)", stage, stage_root)
            continue
        for series_dir in sorted(p for p in stage_root.iterdir() if p.is_dir()):
            series = series_dir.name
            if want and series not in want:
                continue
            entries = discover_samples(series_dir)
            if not entries:
                log.warning("  [%s] %s: no recognizable 10X data found", stage, series)
                continue
            for entry in entries:
                log.info("  inspecting [%s] %s / %s", stage, series, entry["sample"])
                rec = inspect_sample(
                    series, stage, entry,
                    deep=args.deep, max_deep_bytes=args.max_deep_bytes, root=args.root,
                )
                records.append(rec)

    tsv, md = write_reports(records, out_dir)
    log.info("\nWrote %d rows -> %s", len(records), tsv)
    log.info("Report -> %s", md)


if __name__ == "__main__":
    main()
