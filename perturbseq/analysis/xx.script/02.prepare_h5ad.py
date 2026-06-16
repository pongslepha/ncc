#!/usr/bin/env python3
"""Normalize heterogeneous GEO Perturb-seq downloads into the canonical 10X
input that ``perturbseq/model/prepare_perturb_h5ad.py`` consumes, then
(optionally) run that model script to emit the final ``.h5ad``.

Why this script exists
----------------------
``model/prepare_perturb_h5ad.py`` is read-only and strict: it calls
``sc.read_10x_mtx(..., gex_only=False)`` and splits features by the
``feature_types`` column into "Gene Expression" and "CRISPR Guide Capture".
So it requires a SINGLE 10X directory holding one combined matrix plus a
``guide_map.csv``.

The seven GEO series do not all ship that shape. This script reshapes each
into a normalized per-sample directory:

    <out-root>/<GSE>/<sample>/
        barcodes.tsv.gz
        features.tsv.gz      # 3 cols: id, name, feature_type (incl. CRISPR)
        matrix.mtx.gz        # genes x cells, Gene Expression + CRISPR rows
        guide_map.csv        # columns: grna,target_gene  (NT controls -> "NT")

Input shapes handled (auto-selected by --series, override with --shape):
  combined_triple  GSE278572, GSE311503, GSE272457
                   already multi-feature; we re-emit a clean triple + guide_map.
  h5_multifeature  GSE280506
                   read_10x_h5(gex_only=False) -> combined triple + guide_map.
  split_gex_guide  GSE208240
                   separate GEX and gRNA matrices merged on shared barcodes.
  legacy_lookup    GSE142078
                   legacy GEX triple (genes.tsv) + per-cell guide lookup CSV;
                   a CRISPR Guide Capture block is SYNTHESIZED from the lookup.

Usage:
    python 02.prepare_h5ad.py --series GSE311503 \
        --data-root  perturbseq/analysis/00.data \
        --out-root   perturbseq/analysis/00.data/prepared \
        --run-prepare \
        --result-root perturbseq/analysis/01.result
"""
from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io as scipy_io
from scipy import sparse

NT = "NT"  # canonical non-targeting label expected by prepare_perturb_h5ad.py
GUIDE_FEATURE_TYPE = "CRISPR Guide Capture"
GENE_FEATURE_TYPE = "Gene Expression"

MODEL_PREPARE = (
    Path(__file__).resolve().parents[2] / "model" / "prepare_perturb_h5ad.py"
)

# ---------------------------------------------------------------------------
# Per-series rules: input shape, guide-name -> gene parsing, NT detection,
# and any extra args to hand to prepare_perturb_h5ad.py.
# ---------------------------------------------------------------------------
SERIES_RULES: dict[str, dict] = {
    "GSE278572": {
        "shape": "combined_triple",
        "nt_regex": r"(?i)non.?target",
        "strip_regex": r"_\d+_CRISPRi$",
        "mt_pattern": r"^MT-",
        "prepare_args": [],
    },
    "GSE311503": {
        "shape": "combined_triple",
        "nt_regex": r"(?i)no.?target|non.?target",
        "strip_regex": r"[._]\d+$",
        "mt_pattern": r"^MT-",
        "prepare_args": [],
    },
    "GSE272457": {
        "shape": "combined_triple",
        "nt_regex": r"(?i)^nt[._-]",
        "strip_regex": r"_\d+$",
        "mt_pattern": r"^MT-",  # human+mouse mix; symbols stay human-cased
        "prepare_args": [],
    },
    "GSE280506": {
        "shape": "h5_multifeature",
        "nt_regex": r"(?i)non.?target|^nt[._-]|safe.?harbor|scramble",
        "strip_regex": r"[._]\d+$",
        "mt_pattern": r"^MT-",
        "prepare_args": [],
    },
    "GSE208240": {
        "shape": "split_gex_guide",
        "nt_regex": r"(?i)non.?target|^nt[._-]|safe.?harbor|scramble",
        "strip_regex": r"[._-]\d+$",
        "mt_pattern": r"^MT-",
        "prepare_args": [],
    },
    "GSE142078": {
        "shape": "legacy_lookup",
        "nt_regex": r"(?i)^non.?targeting",
        "strip_regex": r"_G?\d+$",
        "mt_pattern": r"^MT-",
        # Synthetic guide counts are 1 UMI/cell, so relax the UMI threshold.
        "prepare_args": ["--min-guide-umi", "1"],
    },
}


# ---------------------------------------------------------------------------
# Guide map derivation
# ---------------------------------------------------------------------------
def derive_gene(guide: str, nt_regex: str, strip_regex: str) -> str:
    """Map a gRNA name to its target gene; NT controls collapse to "NT"."""
    if re.search(nt_regex, guide):
        return NT
    gene = re.sub(strip_regex, "", guide)
    return gene or guide


def build_guide_map(guide_names: list[str], rule: dict) -> pd.DataFrame:
    rows = [
        {"grna": g, "target_gene": derive_gene(g, rule["nt_regex"], rule["strip_regex"])}
        for g in guide_names
    ]
    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    n_nt = int((df["target_gene"] == NT).sum())
    if n_nt == 0:
        print(
            "  [WARN] no non-targeting (NT) guides detected. "
            "prepare_perturb_h5ad.py requires >=1 NT guide. "
            f"Check the nt_regex ({rule['nt_regex']!r}) against these guides: "
            f"{guide_names[:8]}"
        )
    else:
        print(f"  guide_map: {len(df)} guides, {n_nt} NT, "
              f"{df['target_gene'].nunique()} distinct targets (incl. NT).")
    return df


# ---------------------------------------------------------------------------
# 10X writers
# ---------------------------------------------------------------------------
def _write_gzip_text(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")


def write_10x_dir(
    out_dir: Path,
    matrix_genes_x_cells: sparse.spmatrix,
    features: pd.DataFrame,  # columns: id, name, feature_type
    barcodes: list[str],
    guide_map: pd.DataFrame,
) -> None:
    """Write barcodes/features/matrix (gzipped) + guide_map.csv into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_feat, n_cells = matrix_genes_x_cells.shape
    assert n_feat == len(features), (n_feat, len(features))
    assert n_cells == len(barcodes), (n_cells, len(barcodes))

    # matrix.mtx.gz (integer counts, CellRanger orientation: features x cells)
    mtx = matrix_genes_x_cells.tocoo()
    mtx_path = out_dir / "matrix.mtx"
    scipy_io.mmwrite(str(mtx_path), mtx, field="integer")
    with open(mtx_path, "rb") as src, gzip.open(str(mtx_path) + ".gz", "wb") as dst:
        dst.writelines(src)
    mtx_path.unlink()

    # features.tsv.gz : id <tab> name <tab> feature_type
    feat_lines = [
        f"{r.id}\t{r.name}\t{r.feature_type}" for r in features.itertuples(index=False)
    ]
    _write_gzip_text(out_dir / "features.tsv.gz", feat_lines)

    _write_gzip_text(out_dir / "barcodes.tsv.gz", list(barcodes))

    guide_map.to_csv(out_dir / "guide_map.csv", index=False)
    print(f"  wrote 10X dir: {out_dir}  ({n_feat:,} features x {n_cells:,} cells)")


def adata_to_10x(adata: sc.AnnData, out_dir: Path, rule: dict) -> Path:
    """Write a multi-feature AnnData (cells x genes) out as a 10X dir."""
    ft_col = next(
        (c for c in ("feature_types", "feature_type") if c in adata.var.columns),
        None,
    )
    if ft_col is None:
        raise ValueError(
            "AnnData has no feature_types column; cannot identify CRISPR guides."
        )
    feature_types = adata.var[ft_col].astype(str)
    guide_mask = feature_types.eq(GUIDE_FEATURE_TYPE).to_numpy()
    if not guide_mask.any():
        raise ValueError(
            f"No '{GUIDE_FEATURE_TYPE}' features present; this matrix is "
            "gene-expression only and needs the split/lookup path instead."
        )

    gene_ids = (
        adata.var["gene_ids"].astype(str).tolist()
        if "gene_ids" in adata.var.columns
        else adata.var_names.astype(str).tolist()
    )
    features = pd.DataFrame(
        {
            "id": gene_ids,
            "name": adata.var_names.astype(str).tolist(),
            "feature_type": feature_types.tolist(),
        }
    )
    matrix = sparse.csr_matrix(adata.X).T  # genes x cells
    guide_names = adata.var_names[guide_mask].astype(str).tolist()
    guide_map = build_guide_map(guide_names, rule)
    write_10x_dir(out_dir, matrix, features, adata.obs_names.astype(str).tolist(), guide_map)
    return out_dir


# ---------------------------------------------------------------------------
# Shape: combined_triple  (rename messy GEO names to standard 10X names)
# ---------------------------------------------------------------------------
def find_triples(series_dir: Path) -> dict[str, dict[str, Path]]:
    """Group *_barcodes/_features/_matrix files by their sample prefix."""
    samples: dict[str, dict[str, Path]] = {}
    suffix_key = {
        "_barcodes.tsv.gz": "barcodes",
        "_features.tsv.gz": "features",
        "_genes.tsv.gz": "features",  # legacy
        "_matrix.mtx.gz": "matrix",
        "barcodes.tsv.gz": "barcodes",
        "features.tsv.gz": "features",
        "matrix.mtx.gz": "matrix",
    }
    for path in sorted(series_dir.rglob("*")):
        if not path.is_file():
            continue
        for suffix, key in suffix_key.items():
            if path.name.endswith(suffix):
                prefix = path.name[: -len(suffix)].rstrip("_") or path.parent.name
                samples.setdefault(prefix, {})[key] = path
                break
    # keep only complete triples
    return {s: parts for s, parts in samples.items() if {"barcodes", "matrix"} <= parts.keys()}


def load_triple_adata(parts: dict[str, Path]) -> sc.AnnData:
    """Load a triple by staging it under standard 10X names in a temp dir."""
    import tempfile

    staging = Path(tempfile.mkdtemp(prefix="triple_"))
    (staging / "barcodes.tsv.gz").write_bytes(parts["barcodes"].read_bytes())
    (staging / "matrix.mtx.gz").write_bytes(parts["matrix"].read_bytes())
    (staging / "features.tsv.gz").write_bytes(parts["features"].read_bytes())
    adata = sc.read_10x_mtx(staging, var_names="gene_symbols", make_unique=True, gex_only=False)
    return adata


def prep_combined_triple(series_dir: Path, out_root: Path, rule: dict) -> list[Path]:
    triples = find_triples(series_dir)
    if not triples:
        print(f"  [WARN] no complete 10X triples under {series_dir}")
        return []
    out_dirs = []
    for sample, parts in triples.items():
        print(f"  sample '{sample}'")
        adata = load_triple_adata(parts)
        out_dir = out_root / sample
        out_dirs.append(adata_to_10x(adata, out_dir, rule))
    return out_dirs


# ---------------------------------------------------------------------------
# Shape: h5_multifeature
# ---------------------------------------------------------------------------
def prep_h5_multifeature(series_dir: Path, out_root: Path, rule: dict) -> list[Path]:
    h5_files = sorted(series_dir.rglob("*.h5"))
    if not h5_files:
        print(f"  [WARN] no .h5 file under {series_dir}")
        return []
    out_dirs = []
    for h5 in h5_files:
        sample = re.sub(r"_filtered_feature_bc_matrix$", "", h5.stem)
        print(f"  h5 '{h5.name}' -> sample '{sample}'")
        adata = sc.read_10x_h5(h5, gex_only=False)
        adata.var_names_make_unique()
        out_dir = out_root / sample
        out_dirs.append(adata_to_10x(adata, out_dir, rule))
    return out_dirs


# ---------------------------------------------------------------------------
# Shape: split_gex_guide  (GSE208240: separate GEX and guide matrices)
# ---------------------------------------------------------------------------
def _read_any_10x(path: Path) -> sc.AnnData:
    """Read a 10X matrix from a directory or .h5, gex_only=False."""
    if path.is_dir():
        return sc.read_10x_mtx(path, var_names="gene_symbols", make_unique=True, gex_only=False)
    if path.suffix == ".h5":
        a = sc.read_10x_h5(path, gex_only=False)
        a.var_names_make_unique()
        return a
    raise ValueError(f"Don't know how to read 10X data at {path}")


def _classify_matrix_dir(d: Path) -> str:
    """Heuristically label a matrix dir as 'gex' or 'guide' from its path/features."""
    name = d.name.lower() + " " + str(d).lower()
    if "guide" in name or "crispr" in name or "grna" in name or "protospacer" in name:
        return "guide"
    if "gex" in name or "gene" in name or "rna" in name or "expression" in name:
        return "gex"
    return "unknown"


def prep_split_gex_guide(series_dir: Path, out_root: Path, rule: dict) -> list[Path]:
    # Find candidate 10X matrix directories (a dir containing matrix.mtx[.gz]).
    matrix_dirs = []
    for mtx in series_dir.rglob("*matrix.mtx*"):
        matrix_dirs.append(mtx.parent)
    matrix_dirs = sorted(set(matrix_dirs))
    if not matrix_dirs:
        print(f"  [WARN] no matrix.mtx under {series_dir} (did you pass --extract?)")
        return []

    gex_dir = guide_dir = None
    for d in matrix_dirs:
        kind = _classify_matrix_dir(d)
        if kind == "gex" and gex_dir is None:
            gex_dir = d
        elif kind == "guide" and guide_dir is None:
            guide_dir = d
    if gex_dir is None or guide_dir is None:
        print(
            "  [WARN] could not unambiguously identify GEX vs guide matrices.\n"
            f"         candidates: {[str(d) for d in matrix_dirs]}\n"
            "         Inspect the extracted folders and rename/point them so the\n"
            "         GEX dir contains 'gex'/'gene' and the guide dir 'guide'/'crispr'."
        )
        return []

    print(f"  GEX   matrix: {gex_dir}")
    print(f"  guide matrix: {guide_dir}")
    gex = _read_any_10x(gex_dir)
    guide = _read_any_10x(guide_dir)
    return [_merge_gex_guide(gex, guide, out_root / series_dir.name, rule)]


def _merge_gex_guide(
    gex: sc.AnnData, guide: sc.AnnData, out_dir: Path, rule: dict
) -> Path:
    """Concatenate a GEX matrix and a guide matrix on shared cell barcodes."""
    shared = gex.obs_names.intersection(guide.obs_names)
    if len(shared) == 0:
        # barcodes may differ by a "-1" suffix; try stripping
        gex.obs_names = gex.obs_names.str.replace(r"-\d+$", "", regex=True)
        guide.obs_names = guide.obs_names.str.replace(r"-\d+$", "", regex=True)
        shared = gex.obs_names.intersection(guide.obs_names)
    if len(shared) == 0:
        raise ValueError("GEX and guide matrices share no cell barcodes.")
    print(f"  shared cells: {len(shared):,}")
    gex = gex[shared].copy()
    guide = guide[shared].copy()

    gex_ids = (
        gex.var["gene_ids"].astype(str).tolist()
        if "gene_ids" in gex.var.columns
        else gex.var_names.astype(str).tolist()
    )
    guide_names = guide.var_names.astype(str).tolist()
    features = pd.DataFrame(
        {
            "id": gex_ids + guide_names,
            "name": gex.var_names.astype(str).tolist() + guide_names,
            "feature_type": [GENE_FEATURE_TYPE] * gex.n_vars
            + [GUIDE_FEATURE_TYPE] * guide.n_vars,
        }
    )
    matrix = sparse.vstack(
        [sparse.csr_matrix(gex.X).T, sparse.csr_matrix(guide.X).T]
    ).tocsr()  # genes x cells
    guide_map = build_guide_map(guide_names, rule)
    write_10x_dir(out_dir, matrix, features, shared.astype(str).tolist(), guide_map)
    return out_dir


# ---------------------------------------------------------------------------
# Shape: legacy_lookup  (GSE142078: GEX triple + per-cell guide CSV)
# ---------------------------------------------------------------------------
def prep_legacy_lookup(series_dir: Path, out_root: Path, rule: dict) -> list[Path]:
    triples = find_triples(series_dir)
    lookups = sorted(series_dir.rglob("*Cell_Guide_Lookup.csv.gz")) or sorted(
        series_dir.rglob("*[Gg]uide*ookup*.csv*")
    )
    if not triples:
        print(f"  [WARN] no GEX triple under {series_dir}")
        return []
    if not lookups:
        print(f"  [WARN] no *_Cell_Guide_Lookup.csv.gz under {series_dir}")
        return []

    out_dirs = []
    for sample, parts in triples.items():
        # match a lookup file to this sample by shared prefix token (e.g. Run1)
        token = sample.split("_")[-1] if "_" in sample else sample
        lookup = next((l for l in lookups if token in l.name), lookups[0])
        print(f"  sample '{sample}'  + lookup '{lookup.name}'")
        out_dirs.append(
            _build_from_lookup(parts, lookup, out_root / sample, rule)
        )
    return out_dirs


def _build_from_lookup(
    parts: dict[str, Path], lookup_path: Path, out_dir: Path, rule: dict
) -> Path:
    gex = load_triple_adata(parts)  # cells x genes, GEX only (legacy genes.tsv)
    # normalize barcodes (legacy lookups use bare 16bp barcodes, no -1 suffix)
    gex.obs_names = gex.obs_names.str.replace(r"-\d+$", "", regex=True)

    lk = pd.read_csv(lookup_path)
    bc_col = next((c for c in lk.columns if re.search(r"(?i)barcode|cell", c)), lk.columns[0])
    sg_col = next(
        (c for c in lk.columns if re.search(r"(?i)sgrna|guide|grna|gene", c)),
        lk.columns[-1],
    )
    lk = lk[[bc_col, sg_col]].dropna()
    lk.columns = ["barcode", "guide"]
    lk["barcode"] = lk["barcode"].astype(str).str.replace(r"-\d+$", "", regex=True)

    # restrict to cells present in the GEX matrix
    gex_bc = pd.Index(gex.obs_names)
    lk = lk[lk["barcode"].isin(gex_bc)]
    print(f"  lookup covers {lk['barcode'].nunique():,} / {gex.n_obs:,} GEX cells")
    if lk.empty:
        raise ValueError("No lookup barcodes intersect the GEX matrix barcodes.")

    guide_names = sorted(lk["guide"].astype(str).unique())
    guide_index = {g: i for i, g in enumerate(guide_names)}
    bc_pos = {bc: i for i, bc in enumerate(gex.obs_names)}

    # synthesize a guides x cells count matrix: 1 UMI for the assigned guide
    rows, cols = [], []
    for bc, g in zip(lk["barcode"], lk["guide"].astype(str)):
        rows.append(guide_index[g])
        cols.append(bc_pos[bc])
    guide_mat = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int64), (rows, cols)),
        shape=(len(guide_names), gex.n_obs),
    )

    gex_ids = (
        gex.var["gene_ids"].astype(str).tolist()
        if "gene_ids" in gex.var.columns
        else gex.var_names.astype(str).tolist()
    )
    features = pd.DataFrame(
        {
            "id": gex_ids + guide_names,
            "name": gex.var_names.astype(str).tolist() + guide_names,
            "feature_type": [GENE_FEATURE_TYPE] * gex.n_vars
            + [GUIDE_FEATURE_TYPE] * len(guide_names),
        }
    )
    matrix = sparse.vstack([sparse.csr_matrix(gex.X).T, guide_mat]).tocsr()
    guide_map = build_guide_map(guide_names, rule)
    write_10x_dir(out_dir, matrix, features, gex.obs_names.astype(str).tolist(), guide_map)
    return out_dir


# ---------------------------------------------------------------------------
# Optional: invoke model/prepare_perturb_h5ad.py on each normalized dir
# ---------------------------------------------------------------------------
def run_prepare(input_dir: Path, result_root: Path, rule: dict) -> int:
    if not MODEL_PREPARE.exists():
        print(f"  [WARN] {MODEL_PREPARE} not found; skipping prepare step.")
        return 1
    result_root.mkdir(parents=True, exist_ok=True)
    output_prefix = result_root / input_dir.name
    cmd = [
        sys.executable,
        str(MODEL_PREPARE),
        "--input-dir",
        str(input_dir),
        "--output-prefix",
        str(output_prefix),
        "--mt-pattern",
        rule["mt_pattern"],
        *rule.get("prepare_args", []),
    ]
    print("  running:", " ".join(cmd))
    return subprocess.call(cmd)


SHAPE_DISPATCH = {
    "combined_triple": prep_combined_triple,
    "h5_multifeature": prep_h5_multifeature,
    "split_gex_guide": prep_split_gex_guide,
    "legacy_lookup": prep_legacy_lookup,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize GEO Perturb-seq downloads into 10X + guide_map; "
        "optionally run model/prepare_perturb_h5ad.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--series", required=True, help="Comma-separated GEO series IDs.")
    parser.add_argument(
        "--data-root", type=Path, required=True, help="Root of downloaded data (01 --outdir)."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Root for normalized 10X dirs (<out-root>/<GSE>/<sample>/).",
    )
    parser.add_argument(
        "--shape",
        choices=sorted(SHAPE_DISPATCH),
        default=None,
        help="Override the input shape (default: per-series rule).",
    )
    parser.add_argument(
        "--run-prepare",
        action="store_true",
        help="Run model/prepare_perturb_h5ad.py on each normalized dir.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Output root for prepare_perturb_h5ad.py results (with --run-prepare).",
    )
    args = parser.parse_args()

    series_list = [s.strip() for s in args.series.split(",") if s.strip()]
    summary: list[tuple[str, str]] = []

    for series in series_list:
        if series not in SERIES_RULES:
            print(f"\n=== {series} === [WARN] no rule configured; skipping.")
            summary.append((series, "no rule"))
            continue
        rule = SERIES_RULES[series]
        shape = args.shape or rule["shape"]
        series_dir = args.data_root / series
        print(f"\n=== {series} === shape='{shape}'  data={series_dir}")
        if not series_dir.exists():
            print(f"  [WARN] {series_dir} not found; run 01.download_geo.py first.")
            summary.append((series, "no data"))
            continue

        out_root = args.out_root / series
        normalized = SHAPE_DISPATCH[shape](series_dir, out_root, rule)
        if not normalized:
            summary.append((series, "no output"))
            continue

        made_h5ad = 0
        if args.run_prepare:
            result_root = args.result_root or (args.out_root.parent / "01.result" / series)
            for input_dir in normalized:
                rc = run_prepare(input_dir, result_root, rule)
                if rc == 0 and (result_root / f"{input_dir.name}.h5ad").exists():
                    made_h5ad += 1
                else:
                    print(f"  [WARN] prepare failed (rc={rc}) for {input_dir}")
        summary.append(
            (series, f"{len(normalized)} 10X dir(s)" + (f", {made_h5ad} h5ad" if args.run_prepare else ""))
        )

    print("\n=== summary ===")
    for series, status in summary:
        print(f"  {series}: {status}")


if __name__ == "__main__":
    main()
