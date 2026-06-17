#!/usr/bin/env python3
"""Inspect the state of downloaded (raw) and normalized (processed) Perturb-seq
datasets and write a report.

For every dataset/sample it records:
  * stage (raw / processed) and the files present
  * number of cells (barcodes)
  * number of features, split into Gene Expression and CRISPR Guide Capture
  * number of guide RNAs, distinct target genes, and non-targeting (NT) guides
    (from guide_map.csv when available)
  * per-cell guide capture (``--deep``): share of cells with >=1 guide, the
    per-cell guide-UMI fraction (mean/median %) and guides per cell / MOI
    (mean/median) -- guide-UMI fraction is omitted for dial-out samples (no GEX UMIs)
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

import h5py
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
    # GSE157977-style dial-out layout: a GEX-only (legacy CellRanger v2) .h5 per
    # sample plus a separate perturbation-barcode-by-cell "dial-out" count CSV.
    # Guides are not in the h5 (v2 predates feature_types); the PBC->gene map
    # comes from guide_map.csv (built from the paper's Table S5).
    dialouts = sorted(series_dir.rglob("*dialout*Counts.csv*"))
    if dialouts:
        gmap = _find_named(series_dir, "guide_map", suffixes=(".csv",))
        for dcsv in dialouts:
            m = re.search(r"dialout\.([^.]+)", dcsv.name)
            label = m.group(1) if m else dcsv.stem
            h5 = next((p for p in sorted(series_dir.rglob("*.h5"))
                       if f".{label}." in p.name), None)
            entries.append({"sample": label, "kind": "dialout",
                            "dialout": dcsv, "h5": h5, "guide_map": gmap})
        return entries
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


def _h5_gene_count(h5_path: Path) -> int | None:
    """Number of genes in a legacy CellRanger v2 .h5 (genome group with genes)."""
    try:
        with h5py.File(h5_path, "r") as f:
            for k in f.keys():
                grp = f[k]
                if hasattr(grp, "keys"):
                    for key in ("genes", "gene_names"):
                        if key in grp:
                            return int(grp[key].shape[0])
    except Exception:
        return None
    return None


def _mean_guides_per_gene(targets: list[str]) -> float | None:
    """Mean number of guides per *targeting* gene (NT controls excluded).

    ``targets`` is one entry per guide (the guide's target gene). Returns the
    average library coverage, e.g. 1 gene tiled by 4 guides -> 4.0.
    """
    genes = [t for t in targets if t != "NT"]
    if not genes:
        return None
    s = pd.Series(genes)
    return round(float(s.groupby(s).size().mean()), 2)


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


def _per_cell_guide_stats(guide_cells_x_guides, total_umi_per_cell=None) -> dict:
    """Per-cell guide-capture summary over all cells.

    ``guide_cells_x_guides`` is the CRISPR block oriented cells (rows) x guides
    (cols). ``total_umi_per_cell`` (UMI sum across all features) is the
    denominator for the guide-UMI fraction; pass None when GEX UMIs are not
    available (e.g. dial-out), in which case the fraction is left unset.

    Returns pct_cells_with_guide, guide_umi_frac mean/median (%), moi mean/median.
    """
    g = sparse.csr_matrix(guide_cells_x_guides)
    n_cells = g.shape[0]
    if n_cells == 0:
        return {}
    guide_umi = np.asarray(g.sum(axis=1)).ravel().astype(float)
    moi = np.asarray((g > 0).sum(axis=1)).ravel().astype(float)
    out = {
        "pct_cells_with_guide": round(100.0 * float((moi > 0).sum()) / n_cells, 2),
        "moi_mean": round(float(np.mean(moi)), 2),
        "moi_median": round(float(np.median(moi)), 2),
    }
    if total_umi_per_cell is not None:
        total = np.asarray(total_umi_per_cell, dtype=float).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(total > 0, guide_umi / total * 100.0, 0.0)
        out["guide_umi_frac_mean"] = round(float(np.mean(frac)), 2)
        out["guide_umi_frac_median"] = round(float(np.median(frac)), 2)
    return out


def inspect_sample(
    series: str, stage: str, entry: dict, *, deep: bool, max_deep_bytes: int, root: Path
) -> dict:
    sample = entry["sample"]
    rec = {
        "series": series, "stage": stage, "sample": sample,
        "n_cells": None, "n_features": None, "n_gene_expr": None, "n_guides": None,
        "n_targets": None, "n_nt": None, "guides_per_gene": None,
        "pct_cells_with_guide": None,
        "guide_umi_frac_mean": None, "guide_umi_frac_median": None,
        "moi_mean": None, "moi_median": None,
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
            rec["guides_per_gene"] = _mean_guides_per_gene(
                df["target_gene"].astype(str).tolist()
            )
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
                    guide_X = adata[:, gmask].X
                    dom = _guide_dominance_from_matrix(
                        guide_X, adata.var_names[gmask].astype(str).tolist()
                    )
                    rec.update(dom)
                    total_umi = np.asarray(
                        sparse.csr_matrix(adata.X).sum(axis=1)).ravel()
                    rec.update(_per_cell_guide_stats(guide_X, total_umi))

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
                per_guide_target = [_target_from_guide(c) for c in gcols]
                rec["n_targets"] = len(set(per_guide_target))
                rec["n_nt"] = sum(1 for c in gcols if re.match(r"(?i)^neg", str(c)))
                rec["guides_per_gene"] = _mean_guides_per_gene(per_guide_target)
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
                guide_X = adata[:, gmask].X
                dom = _guide_dominance_from_matrix(
                    guide_X, adata.var_names[gmask].astype(str).tolist()
                )
                rec.update(dom)
                total_umi = np.asarray(
                    sparse.csr_matrix(adata.X).sum(axis=1)).ravel()
                rec.update(_per_cell_guide_stats(guide_X, total_umi))
        else:
            notes.append("h5 has no feature_types column")

    elif entry["kind"] == "dialout":
        # GEX gene count from the paired (GEX-only) h5 -- header read, no matrix load.
        if entry.get("h5") is not None:
            n_g = _h5_gene_count(entry["h5"])
            if n_g:
                rec["n_features"] = rec["n_gene_expr"] = n_g
        # library (PBC -> gene) from the Table S5-derived guide_map.csv.
        pbc2gene: dict[str, str] = {}
        gmap = entry.get("guide_map")
        if gmap is not None and gmap.exists():
            gm = pd.read_csv(gmap)
            rec["n_guides"] = len(gm)
            rec["n_targets"] = int(gm["target_gene"].nunique())
            rec["n_nt"] = int((gm["target_gene"] == "NT").sum())
            rec["guides_per_gene"] = _mean_guides_per_gene(
                gm["target_gene"].astype(str).tolist()
            )
            if "perturbation_barcode" in gm.columns:
                pbc2gene = dict(zip(gm["perturbation_barcode"].astype(str),
                                    gm["target_gene"].astype(str)))
            notes.append("library = Table S5 (2 sgRNAs/gene, 1 dial-out barcode/gene, "
                         "GFP control mapped to NT)")
        else:
            notes.append("no guide_map.csv (Table S5) found; PBCs unmapped")
        # dial-out matrix: cells with a perturbation call (+ dominance over PBCs).
        dial = pd.read_csv(entry["dialout"], index_col=0)
        rec["n_cells"] = int(dial.shape[0])
        cols = [str(c) for c in dial.columns]
        n_detected = len(cols)
        n_unknown = sum(1 for c in cols if pbc2gene and c not in pbc2gene)
        note = f"{n_detected} perturbation barcodes detected in this sample"
        if n_unknown:
            note += f" ({n_unknown} not in Table S5)"
        notes.append(note)
        if deep and n_detected:
            rec.update(_per_cell_guide_stats(sparse.csr_matrix(dial.to_numpy())))
            colsum = dial.sum(axis=0).astype(float)
            ubiq = (dial > 0).sum(axis=0) / max(1, dial.shape[0])
            total = float(colsum.sum())
            if total > 0:
                top = str(colsum.idxmax())
                gene = pbc2gene.get(top, "?")
                rec["top_guide"] = f"{gene} [{top}]"
                rec["top_guide_umi_frac"] = round(float(colsum[top]) / total, 4)
                rec["top_guide_ubiquity"] = round(float(ubiq[top]), 4)

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
                cap = ""
                if r.get("pct_cells_with_guide") is not None:
                    parts_cap = [f"cells w/ guide={_fmt_num(r['pct_cells_with_guide'])}%"]
                    if r.get("guide_umi_frac_mean") is not None:
                        parts_cap.append(
                            f"guide UMI%/cell mean={_fmt_num(r['guide_umi_frac_mean'])} "
                            f"median={_fmt_num(r['guide_umi_frac_median'])}")
                    parts_cap.append(
                        f"MOI mean={_fmt_num(r['moi_mean'])} "
                        f"median={_fmt_num(r['moi_median'])}")
                    cap = "  \n    guide capture: " + ", ".join(parts_cap)
                lines.append(
                    f"- **[{r['stage']}] {r['sample']}** — "
                    f"cells={_fmt(r['n_cells'])}, features={_fmt(r['n_features'])} "
                    f"(GEX={_fmt(r['n_gene_expr'])}, guides={_fmt(r['n_guides'])}), "
                    f"targets={_fmt(r['n_targets'])}, NT={_fmt(r['n_nt'])}, "
                    f"guides/gene={_fmt_num(r['guides_per_gene'])}{dom}{cap}"
                )
                if r["notes"]:
                    lines.append(f"    - notes: {r['notes']}")
            lines.append("")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tsv, md


def _fmt(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{int(v):,}"


def _fmt_num(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):g}"


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
    parser.add_argument("--which", choices=["raw", "processed", "both", "auto"], default="auto",
                        help="Which stage(s) to inspect. 'auto' (default) reports processed "
                             "samples, falling back to raw only for series that have no "
                             "processed dir (e.g. GSE278572, too large to process here).")
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

    processed_root = geo_utils.processed_root(args.root)
    # Series that have a processed dir -- in 'auto' mode these are taken from
    # processed only, and raw is consulted just for series missing from here.
    processed_series = (
        {p.name for p in processed_root.iterdir() if p.is_dir()}
        if processed_root.exists() else set()
    )

    stage_roots = []
    if args.which in ("raw", "both", "auto"):
        stage_roots.append(("raw", geo_utils.raw_root(args.root)))
    if args.which in ("processed", "both", "auto"):
        stage_roots.append(("processed", processed_root))

    records: list[dict] = []
    for stage, stage_root in stage_roots:
        if not stage_root.exists():
            log.info("(%s root not found: %s)", stage, stage_root)
            continue
        for series_dir in sorted(p for p in stage_root.iterdir() if p.is_dir()):
            series = series_dir.name
            if want and series not in want:
                continue
            # auto: only fall back to raw for series with no processed dir
            if args.which == "auto" and stage == "raw" and series in processed_series:
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
