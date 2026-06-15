"""
Build a scanpy-preprocessed h5ad for model_a_perturbation_transition_pytorch.py
from a 10X Perturb-seq directory + a user-provided guide_map.csv.

Expected input directory layout:
    <input_dir>/
        barcodes.tsv(.gz)
        features.tsv(.gz)
        matrix.mtx(.gz)
        guide_map.csv      # columns: (grna, target_gene) OR (guide_id, gene)
                           # non-targeting guides MUST have gene == "NT"

Output (paths derived from --output-prefix):
    <output_prefix>.h5ad
    <output_prefix>_qc_violin_before_filter.png
    <output_prefix>_nt_guide_violin.png
    <output_prefix>_qc_violin_by_gene_before_mixscape.png
    <output_prefix>_umap_qc_panels.png

Usage:
    python make_h5ad.py --input-dir D:/.../GSE213921 --output-prefix scanpy_perturb_obj
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Required packages:
# scanpy anndata scipy pandas numpy seaborn matplotlib python-igraph leidenalg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse


GUIDE_MAP_FILENAME = "guide_map.csv"
NT_TARGET_NAME = "NT"

# Defaults aligned with the previous Seurat/mixscape-style pipeline.
DEFAULTS = {
    "min_nfeature_rna": 1000,
    "min_guide_umi": 5,
    "max_guides_detected": 1,
    "guide_detection_umi": 1,
    "n_hvg": 3000,
    "n_pcs": 50,
    "n_neighbor_pcs": 30,
    "cluster_resolution": 0.4,
    # "^MT-" matches human gene symbols; pass "^mt-" for mouse.
    "mt_pattern": r"^MT-",
}


# ---------------------------------------------------------------------------
# Guide map loading
# ---------------------------------------------------------------------------

def load_guide_map(input_dir: Path) -> pd.DataFrame:
    """Load the user-supplied guide_map.csv and normalize columns.

    Returns a DataFrame with columns (guide_id, gene). NT controls are
    identified by gene == "NT".
    """
    path = input_dir / GUIDE_MAP_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"guide_map.csv not found at {path}. Provide a CSV with columns "
            "(grna, target_gene) or (guide_id, gene); NT controls must use "
            f"gene == '{NT_TARGET_NAME}'."
        )

    df = pd.read_csv(path)
    cols = set(df.columns)

    if {"grna", "target_gene"}.issubset(cols):
        df = df.rename(columns={"grna": "guide_id", "target_gene": "gene"})
    elif {"guide_id", "gene"}.issubset(cols):
        pass
    else:
        raise ValueError(
            f"{path} must contain columns (grna, target_gene) or "
            f"(guide_id, gene). Got: {sorted(cols)}"
        )

    df = df[["guide_id", "gene"]].astype(str).drop_duplicates()
    df["guide_id"] = df["guide_id"].str.strip()
    df["gene"] = df["gene"].str.strip()

    n_nt = int((df["gene"] == NT_TARGET_NAME).sum())
    if n_nt == 0:
        raise ValueError(
            f"No non-targeting controls found in {path}. At least one row "
            f"must have gene == '{NT_TARGET_NAME}'."
        )

    print(f"Loaded {len(df)} guides from {path} ({n_nt} NT controls).")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 10X reading
# ---------------------------------------------------------------------------

def _feature_type_column(adata: sc.AnnData) -> str:
    for col in ("feature_types", "feature_type"):
        if col in adata.var.columns:
            return col
    raise ValueError(
        "Could not find a feature type column in adata.var. "
        "Expected 'feature_types' from a 10X multi-feature matrix."
    )


def read_10x_rna_and_guides(data_dir: Path) -> sc.AnnData:
    """Read a 10X multi-feature directory; keep RNA in X, gRNA in obsm."""
    adata_all = sc.read_10x_mtx(
        data_dir,
        var_names="gene_symbols",
        make_unique=True,
        gex_only=False,
    )

    feature_type_col = _feature_type_column(adata_all)
    feature_types = adata_all.var[feature_type_col].astype(str)

    rna_mask = feature_types.eq("Gene Expression").to_numpy()
    guide_mask = feature_types.eq("CRISPR Guide Capture").to_numpy()

    if not rna_mask.any():
        raise ValueError("No 'Gene Expression' features were found.")
    if not guide_mask.any():
        raise ValueError("No 'CRISPR Guide Capture' features were found.")

    adata = adata_all[:, rna_mask].copy()
    guide_adata = adata_all[:, guide_mask].copy()

    adata.layers["counts"] = adata.X.copy()
    adata.obsm["gRNA_counts"] = guide_adata.X.copy().tocsr()
    adata.uns["gRNA_names"] = guide_adata.var_names.astype(str).to_numpy()
    adata.uns["gRNA_var"] = guide_adata.var.copy()

    adata.uns["project"] = "PerturbSeq"
    return adata


# ---------------------------------------------------------------------------
# QC and plotting
# ---------------------------------------------------------------------------

def add_rna_qc(adata: sc.AnnData, mt_pattern: str) -> None:
    adata.var["mt"] = adata.var_names.astype(str).str.match(mt_pattern)
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    adata.obs["nFeature_RNA"] = adata.obs["n_genes_by_counts"]
    adata.obs["nCount_RNA"] = adata.obs["total_counts"]
    adata.obs["percent.mt"] = adata.obs["pct_counts_mt"]


def save_obs_violin(
    adata: sc.AnnData,
    columns: list[str],
    out_path: Path,
    groupby: str | None = None,
) -> None:
    if adata.n_obs == 0:
        print(f"Skipping {out_path.name}: adata has 0 cells.")
        return
    if groupby is not None and adata.obs[groupby].nunique() == 0:
        print(f"Skipping {out_path.name}: no values in groupby column '{groupby}'.")
        return

    if groupby is None:
        plot_df = adata.obs[columns].melt(var_name="feature", value_name="value")
        plt.figure(figsize=(4 * len(columns), 4))
        sns.violinplot(data=plot_df, x="feature", y="value", cut=0, inner=None)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return

    plot_df = adata.obs[[groupby, *columns]].melt(
        id_vars=groupby,
        var_name="feature",
        value_name="value",
    )
    g = sns.catplot(
        data=plot_df,
        x=groupby,
        y="value",
        col="feature",
        kind="violin",
        cut=0,
        inner=None,
        sharey=False,
        height=4,
        aspect=1.3,
    )
    g.set_xticklabels(rotation=90)
    g.figure.tight_layout()
    g.figure.savefig(out_path, dpi=200)
    plt.close(g.figure)


def save_guide_violin(
    adata: sc.AnnData, guide_names: list[str], out_path: Path
) -> None:
    all_guides = pd.Index(adata.uns["gRNA_names"].astype(str))
    existing_guides = [guide for guide in guide_names if guide in all_guides]
    missing_guides = sorted(set(guide_names) - set(existing_guides))

    if missing_guides:
        print(f"Warning: missing NT guides in guide matrix: {missing_guides}")
    if not existing_guides:
        print("Skipping NT guide violin plot — no requested guides were found.")
        return

    guide_idx = all_guides.get_indexer(existing_guides)
    guide_counts = adata.obsm["gRNA_counts"][:, guide_idx]
    guide_df = pd.DataFrame.sparse.from_spmatrix(
        guide_counts,
        index=adata.obs_names,
        columns=existing_guides,
    )
    plot_df = guide_df.melt(var_name="guide", value_name="count")

    plt.figure(figsize=(6, 4))
    sns.violinplot(data=plot_df, x="guide", y="count", cut=0, inner=None)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Guide calling
# ---------------------------------------------------------------------------

def sparse_row_max_and_argmax(
    matrix: sparse.spmatrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row-wise max values and argmax indices without densifying."""
    matrix = matrix.tocsr()
    row_max = np.zeros(matrix.shape[0], dtype=float)
    row_argmax = np.zeros(matrix.shape[0], dtype=np.int64)

    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        if start == end:
            continue

        values = matrix.data[start:end]
        cols = matrix.indices[start:end]
        positive = values > 0
        if not positive.any():
            continue

        positive_values = values[positive]
        positive_cols = cols[positive]
        best = int(np.argmax(positive_values))
        row_max[row] = float(positive_values[best])
        row_argmax[row] = int(positive_cols[best])

    return row_max, row_argmax


def add_guide_calls(
    adata: sc.AnnData,
    guide_map: pd.DataFrame,
    min_guide_umi: int,
    max_guides_detected: int,
    guide_detection_umi: int = 1,
) -> None:
    """Assign a dominant guide and target gene per cell using guide_map.

    A guide counts as 'detected' in a cell only if its UMI >=
    guide_detection_umi (default 1 = any nonzero count). Raising this
    helps with ambient-guide contamination in CROPseq-style datasets.
    """
    guide_counts = adata.obsm["gRNA_counts"].tocsr()
    guide_names = pd.Index(adata.uns["gRNA_names"].astype(str))

    top_umi, top_idx = sparse_row_max_and_argmax(guide_counts)
    n_detected_guides = np.asarray(
        (guide_counts >= guide_detection_umi).sum(axis=1)
    ).ravel()

    top_guide = np.full(adata.n_obs, pd.NA, dtype=object)
    has_guide = top_umi > 0
    top_guide[has_guide] = guide_names.to_numpy()[top_idx[has_guide]]

    adata.obs["guide_id"] = top_guide
    adata.obs["guide_umi"] = top_umi
    adata.obs["n_guides_detected"] = n_detected_guides.astype(int)

    high_confidence = (
        adata.obs["guide_id"].notna()
        & (adata.obs["guide_umi"] >= min_guide_umi)
        & (adata.obs["n_guides_detected"] <= max_guides_detected)
    )

    adata.obs["guide_call"] = "unassigned"
    adata.obs.loc[high_confidence, "guide_call"] = adata.obs.loc[
        high_confidence, "guide_id"
    ].astype(str)

    guide_to_gene = guide_map.set_index("guide_id")["gene"]

    matrix_guides = set(guide_names)
    map_guides = set(guide_to_gene.index)
    overlap = matrix_guides & map_guides
    only_in_matrix = matrix_guides - map_guides
    only_in_map = map_guides - matrix_guides

    if len(overlap) == 0:
        raise ValueError(
            "guide_map.csv does not match the gRNA names in the count matrix: "
            f"0 of {len(matrix_guides)} matrix gRNAs are present in the map "
            f"({len(map_guides)} entries).\n"
            f"  Example matrix gRNA names (uns['gRNA_names']): "
            f"{sorted(matrix_guides)[:5]}\n"
            f"  Example guide_map.csv guide_id values:          "
            f"{sorted(map_guides)[:5]}\n"
            "Check that the 'grna'/'guide_id' column in guide_map.csv uses the "
            "exact same identifiers as the CRISPR Guide Capture features."
        )

    print(
        f"guide_map.csv overlap: {len(overlap)}/{len(matrix_guides)} "
        f"matrix gRNAs are mapped "
        f"({100 * len(overlap) / len(matrix_guides):.1f}%)."
    )
    if only_in_matrix:
        print(
            f"Warning: {len(only_in_matrix)} gRNAs in the matrix have no "
            f"entry in guide_map.csv; their cells will be labeled 'unknown'. "
            f"Example: {sorted(only_in_matrix)[:5]}"
        )
    if only_in_map:
        print(
            f"Note: {len(only_in_map)} guide_id values in guide_map.csv are "
            f"not present in the matrix and will be ignored. "
            f"Example: {sorted(only_in_map)[:5]}"
        )

    adata.obs["gene"] = (
        adata.obs["guide_call"].map(guide_to_gene).fillna("unknown")
    )


# ---------------------------------------------------------------------------
# Expression preprocessing
# ---------------------------------------------------------------------------

def preprocess_rna(adata: sc.AnnData, params: dict) -> None:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=params["n_hvg"],
            flavor="seurat_v3",
            layer="counts",
        )
    except ImportError:
        print(
            "scikit-misc is not installed; falling back to Scanpy's 'seurat' "
            "HVG flavor."
        )
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=params["n_hvg"],
            flavor="seurat",
        )

    sc.pp.regress_out(adata, keys=["nCount_RNA", "percent.mt"])
    sc.pp.scale(adata, max_value=10)

    n_hvg_selected = int(adata.var["highly_variable"].sum())
    max_pcs = max(1, min(adata.n_obs - 1, n_hvg_selected - 1))
    n_pcs = min(params["n_pcs"], max_pcs)
    if n_pcs < params["n_pcs"]:
        print(
            f"Warning: requested --n-pcs={params['n_pcs']} exceeds "
            f"min(n_cells-1, n_hvg-1)={max_pcs} "
            f"(n_cells={adata.n_obs}, n_hvg={n_hvg_selected}); using n_pcs={n_pcs}."
        )

    n_neighbor_pcs = min(params["n_neighbor_pcs"], n_pcs)
    if n_neighbor_pcs < params["n_neighbor_pcs"]:
        print(
            f"Warning: requested --n-neighbor-pcs={params['n_neighbor_pcs']} "
            f"exceeds available PCs={n_pcs}; using n_neighbor_pcs={n_neighbor_pcs}."
        )

    sc.tl.pca(
        adata,
        n_comps=n_pcs,
        use_highly_variable=True,
        svd_solver="arpack",
    )
    sc.pp.neighbors(adata, n_pcs=n_neighbor_pcs)

    sc.tl.leiden(
        adata,
        resolution=params["cluster_resolution"],
        key_added="seurat_clusters_raw",
    )
    sc.tl.umap(adata)

    adata.obs["seurat_clusters"] = (
        "C" + adata.obs["seurat_clusters_raw"].astype(str).astype(object)
    )


def save_umap_plots(adata: sc.AnnData, out_path: Path) -> None:
    adata.obs["is_nt"] = np.where(
        adata.obs["gene"].eq(NT_TARGET_NAME), NT_TARGET_NAME, "Target_gene"
    )

    sc.pl.umap(
        adata,
        color=["seurat_clusters", "is_nt", "gene"],
        ncols=3,
        size=8,
        frameon=False,
        show=False,
        wspace=0.4,
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------

def resolve_output_prefix(output_prefix: str, input_dir: Path) -> Path:
    """A bare prefix is placed in input_dir; a path-like prefix is used as-is."""
    prefix = Path(output_prefix)
    if prefix.parent == Path(""):
        prefix = input_dir / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return prefix


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_h5ad(
    input_dir: Path,
    output_prefix: Path,
    *,
    min_nfeature_rna: int = DEFAULTS["min_nfeature_rna"],
    min_guide_umi: int = DEFAULTS["min_guide_umi"],
    max_guides_detected: int = DEFAULTS["max_guides_detected"],
    guide_detection_umi: int = DEFAULTS["guide_detection_umi"],
    n_hvg: int = DEFAULTS["n_hvg"],
    n_pcs: int = DEFAULTS["n_pcs"],
    n_neighbor_pcs: int = DEFAULTS["n_neighbor_pcs"],
    cluster_resolution: float = DEFAULTS["cluster_resolution"],
    mt_pattern: str = DEFAULTS["mt_pattern"],
    sample: str = "onecond.",
    batch: str = "batch1",
    condition: str = "condition",
) -> Path:
    sc.settings.verbosity = 2
    sns.set_theme(style="whitegrid")

    params = {
        "n_hvg": n_hvg,
        "n_pcs": n_pcs,
        "n_neighbor_pcs": n_neighbor_pcs,
        "cluster_resolution": cluster_resolution,
    }

    guide_map = load_guide_map(input_dir)
    nt_guides = guide_map.loc[
        guide_map["gene"] == NT_TARGET_NAME, "guide_id"
    ].tolist()

    adata = read_10x_rna_and_guides(input_dir)
    add_rna_qc(adata, mt_pattern=mt_pattern)

    save_obs_violin(
        adata,
        ["nFeature_RNA", "nCount_RNA", "percent.mt"],
        output_prefix.with_name(output_prefix.name + "_qc_violin_before_filter.png"),
    )

    adata = adata[adata.obs["nFeature_RNA"] > min_nfeature_rna].copy()

    save_guide_violin(
        adata,
        nt_guides,
        output_prefix.with_name(output_prefix.name + "_nt_guide_violin.png"),
    )

    adata.obs["sample"] = sample
    adata.obs["batch"] = batch
    adata.obs["condition"] = condition

    add_guide_calls(
        adata,
        guide_map,
        min_guide_umi=min_guide_umi,
        max_guides_detected=max_guides_detected,
        guide_detection_umi=guide_detection_umi,
    )

    adata = adata[
        (adata.obs["gene"] != "unknown")
        & (adata.obs["guide_call"] != "unassigned")
    ].copy()

    print("\nCells per target gene:")
    print(adata.obs["gene"].value_counts().sort_index())
    print("\nCells per guide:")
    print(adata.obs["guide_call"].value_counts().sort_index())

    save_obs_violin(
        adata,
        ["nFeature_RNA", "nCount_RNA", "guide_umi"],
        output_prefix.with_name(
            output_prefix.name + "_qc_violin_by_gene_before_mixscape.png"
        ),
        groupby="gene",
    )

    preprocess_rna(adata, params)
    save_umap_plots(
        adata,
        output_prefix.with_name(output_prefix.name + "_umap_qc_panels.png"),
    )

    h5ad_path = output_prefix.with_suffix(".h5ad")
    adata.write_h5ad(h5ad_path)
    print(f"\nSaved: {h5ad_path}")
    print(f"Final shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return h5ad_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a scanpy-preprocessed h5ad for "
            "model_a_perturbation_transition_pytorch.py from a 10X Perturb-seq "
            "directory + user-provided guide_map.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["g", "h"],
        default="h",
        help=(
            "Operation mode. 'h' (default): build a scanpy-preprocessed h5ad. "
            "'g': load only the 10X data (barcodes/features/matrix) and write "
            "the list of CRISPR Guide Capture gRNA names, then exit. "
            "guide_map.csv is NOT required in 'g' mode."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory holding 10X files (barcodes/features/matrix). "
            "Mode 'h' also requires guide_map.csv (columns: grna,target_gene "
            "OR guide_id,gene; NT controls labeled 'NT')."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help=(
            "Output prefix. Bare prefix is placed inside --input-dir; an "
            "absolute or path-like prefix is used as-is. "
            "Mode 'h' produces <prefix>.h5ad and <prefix>_*.png. "
            "Mode 'g' produces <prefix>_grna_names.txt."
        ),
    )
    parser.add_argument(
        "--min-nfeature-rna",
        type=int,
        default=DEFAULTS["min_nfeature_rna"],
        help="Drop cells with nFeature_RNA <= this value (RNA QC filter).",
    )
    parser.add_argument(
        "--min-guide-umi",
        type=int,
        default=DEFAULTS["min_guide_umi"],
        help="Minimum UMI count for the dominant gRNA to assign a high-confidence guide_call.",
    )
    parser.add_argument(
        "--max-guides-detected",
        type=int,
        default=DEFAULTS["max_guides_detected"],
        help=(
            "Maximum number of distinct gRNAs detected (UMI >= "
            "--guide-detection-umi) per cell to still assign a high-confidence "
            "guide_call. 1 = single-guide cells only; raise to allow "
            "multi-guide cells."
        ),
    )
    parser.add_argument(
        "--guide-detection-umi",
        type=int,
        default=DEFAULTS["guide_detection_umi"],
        help=(
            "Minimum UMI count for a guide to count as 'detected' in a cell "
            "when computing n_guides_detected. Default 1 (any nonzero count). "
            "Raise (e.g. 3-5) for CROPseq-style datasets with ambient guide "
            "contamination."
        ),
    )
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=DEFAULTS["n_hvg"],
        help="Number of highly variable genes to select.",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=DEFAULTS["n_pcs"],
        help="Number of principal components to compute (sc.tl.pca n_comps).",
    )
    parser.add_argument(
        "--n-neighbor-pcs",
        type=int,
        default=DEFAULTS["n_neighbor_pcs"],
        help="Number of PCs to use when building the neighborhood graph (sc.pp.neighbors n_pcs).",
    )
    parser.add_argument(
        "--cluster-resolution",
        type=float,
        default=DEFAULTS["cluster_resolution"],
        help="Resolution for Leiden clustering (sc.tl.leiden).",
    )
    parser.add_argument(
        "--mt-pattern",
        type=str,
        default=DEFAULTS["mt_pattern"],
        help="Regex for mitochondrial gene symbols (use '^mt-' for mouse).",
    )
    return parser.parse_args()


def extract_grna_names(input_dir: Path, output_prefix: Path) -> Path:
    """Mode 'g': load 10X data and write the gRNA name list only."""
    sc.settings.verbosity = 2
    adata = read_10x_rna_and_guides(input_dir)
    grna_names = [str(name) for name in adata.uns["gRNA_names"]]

    out_path = output_prefix.with_name(output_prefix.name + "_grna_names.txt")
    out_path.write_text("\n".join(grna_names) + "\n", encoding="utf-8")

    print(f"Cells loaded: {adata.n_obs:,}")
    print(f"Found {len(grna_names)} CRISPR Guide Capture features.")
    print(f"Wrote: {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir is not a directory: {input_dir}")

    output_prefix = resolve_output_prefix(args.output_prefix, input_dir)

    if args.mode == "g":
        extract_grna_names(input_dir, output_prefix)
        return

    build_h5ad(
        input_dir=input_dir,
        output_prefix=output_prefix,
        min_nfeature_rna=args.min_nfeature_rna,
        min_guide_umi=args.min_guide_umi,
        max_guides_detected=args.max_guides_detected,
        guide_detection_umi=args.guide_detection_umi,
        n_hvg=args.n_hvg,
        n_pcs=args.n_pcs,
        n_neighbor_pcs=args.n_neighbor_pcs,
        cluster_resolution=args.cluster_resolution,
        mt_pattern=args.mt_pattern,
    )


if __name__ == "__main__":
    main()
