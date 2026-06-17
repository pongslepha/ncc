#!/usr/bin/env python3
"""Verify that every processed sample carries a proper gRNA x cell CRISPR count
matrix, and print a 5x5 corner of each to a log.

For each <processed>/<GSE>/<sample>/ 10X dir this:
  * reads the matrix (gex_only=False) and splits features by feature_types,
  * extracts the CRISPR Guide Capture block oriented as gRNA (rows) x cell (cols),
  * sanity-checks it (block present, integer counts, dims match guide_map),
  * reports per-cell guide capture: share of cells with >=1 guide, the per-cell
    guide-UMI fraction (mean/median %), and guides per cell / MOI (mean/median),
  * prints matrix[:5, :5] with guide names as the index and cell barcodes as cols.

Output goes to <root>/logs/guide_matrix_check.log (and stdout).

Usage:
    python 04.check_guide_matrix.py --root perturbseq/analysis/00.data
    python 04.check_guide_matrix.py --root perturbseq/analysis/00.data --series GSE157977
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import geo_utils

GUIDE_FT = "CRISPR Guide Capture"
GENE_FT = "Gene Expression"


def _has_matrix(d: Path) -> bool:
    return any((d / f"matrix.mtx{ext}").exists() for ext in ("", ".gz"))


def _per_cell_guide_stats(guide_cells_x_guides, total_umi_per_cell) -> dict:
    """Per-cell guide-capture summary statistics.

    ``guide_cells_x_guides`` is the CRISPR block oriented cells (rows) x guides
    (cols); ``total_umi_per_cell`` is each cell's UMI sum across *all* features
    (GEX + guide), used as the denominator for the guide-UMI fraction.

    Returns, over all cells:
      * pct_cells_with_guide -- share of cells with >=1 guide UMI detected
      * guide_umi_frac mean/median -- per-cell 100 * guide_UMIs / total_UMIs (%)
      * moi mean/median -- per-cell number of distinct guides detected
    """
    g = sparse.csr_matrix(guide_cells_x_guides)
    n_cells = g.shape[0]
    guide_umi = np.asarray(g.sum(axis=1)).ravel().astype(float)
    moi = np.asarray((g > 0).sum(axis=1)).ravel().astype(float)
    total = np.asarray(total_umi_per_cell, dtype=float).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total > 0, guide_umi / total * 100.0, 0.0)
    has_guide = moi > 0
    return {
        "n_cells": n_cells,
        "n_cells_with_guide": int(has_guide.sum()),
        "pct_cells_with_guide": 100.0 * has_guide.sum() / max(1, n_cells),
        "guide_umi_frac_mean": float(np.mean(frac)) if n_cells else 0.0,
        "guide_umi_frac_median": float(np.median(frac)) if n_cells else 0.0,
        "moi_mean": float(np.mean(moi)) if n_cells else 0.0,
        "moi_median": float(np.median(moi)) if n_cells else 0.0,
    }


def check_sample(series: str, sample_dir: Path, lines: list[str]) -> None:
    sample = sample_dir.name
    head = f"=== {series} / {sample} ==="
    adata = sc.read_10x_mtx(sample_dir, var_names="gene_symbols",
                            make_unique=True, gex_only=False)
    ft_col = next((c for c in ("feature_types", "feature_type")
                   if c in adata.var.columns), None)
    if ft_col is None:
        lines.append(f"{head}\n  [FAIL] no feature_types column in features.tsv")
        return
    ft = adata.var[ft_col].astype(str)
    gmask = ft.eq(GUIDE_FT).to_numpy()
    n_guides, n_gex = int(gmask.sum()), int(ft.eq(GENE_FT).sum())
    if not gmask.any():
        lines.append(f"{head}\n  [FAIL] no '{GUIDE_FT}' features (GEX-only matrix)")
        return

    # gRNA (rows) x cell (cols)
    guide = adata[:, gmask]
    mat = sparse.csr_matrix(guide.X).T            # guides x cells
    guide_names = guide.var_names.astype(str).tolist()
    barcodes = adata.obs_names.astype(str).tolist()

    # per-cell guide-capture stats (denominator = total UMIs over all features)
    total_umi = np.asarray(sparse.csr_matrix(adata.X).sum(axis=1)).ravel()
    stats = _per_cell_guide_stats(guide.X, total_umi)

    # sanity checks
    data = mat.data
    is_int = bool(np.all(data == np.round(data))) if data.size else True
    gm_path = sample_dir / "guide_map.csv"
    gm_note = ""
    if gm_path.exists():
        gm = pd.read_csv(gm_path)
        mapped = len(set(guide_names) & set(gm["grna"].astype(str)))
        gm_note = (f", guide_map={len(gm)} rows, "
                   f"{mapped}/{n_guides} matrix gRNAs mapped")
    status = "OK" if is_int else "WARN (non-integer counts)"

    lines.append(head)
    lines.append(f"  [{status}] gRNA x cell = {n_guides} x {len(barcodes)}  "
                 f"(GEX features={n_gex}; total nnz in guide block={mat.nnz:,}{gm_note})")

    # per-cell guide-capture summary
    lines.append(
        f"  cells with >=1 guide: {stats['n_cells_with_guide']:,}/{stats['n_cells']:,} "
        f"({stats['pct_cells_with_guide']:.1f}%)")
    lines.append(
        f"  guide UMI fraction per cell: mean={stats['guide_umi_frac_mean']:.2f}%, "
        f"median={stats['guide_umi_frac_median']:.2f}%")
    lines.append(
        f"  guides per cell (MOI): mean={stats['moi_mean']:.2f}, "
        f"median={stats['moi_median']:.1f}")

    # 5x5 corner
    r, c = min(5, n_guides), min(5, len(barcodes))
    corner = mat[:r, :c].toarray()
    df = pd.DataFrame(corner,
                      index=[g[:22] for g in guide_names[:r]],
                      columns=[b[:18] for b in barcodes[:c]])
    df.index.name = "gRNA \\ cell"
    lines.append("  matrix[1:5, 1:5] (gRNA rows x cell cols):")
    lines.append("\n".join("    " + ln for ln in df.to_string().splitlines()))
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="Data root containing processed/.")
    ap.add_argument("--series", default=None,
                    help="Comma-separated GEO IDs (default: all processed).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Log path (default: <root>/logs/guide_matrix_check.log).")
    args = ap.parse_args()

    proc = geo_utils.processed_root(args.root)
    out = args.out or (geo_utils.logs_root(args.root) / "guide_matrix_check.log")
    want = {s.strip() for s in args.series.split(",")} if args.series else None

    lines: list[str] = ["# Processed guide (gRNA x cell) count-matrix check", ""]
    for series_dir in sorted(p for p in proc.iterdir() if p.is_dir()):
        if want and series_dir.name not in want:
            continue
        sample_dirs = sorted(d for d in series_dir.rglob("*")
                             if d.is_dir() and _has_matrix(d))
        for sd in sample_dirs:
            try:
                check_sample(series_dir.name, sd, lines)
            except Exception as exc:  # noqa: BLE001
                lines.append(f"=== {series_dir.name} / {sd.name} ===\n  [ERROR] {exc}\n")

    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
