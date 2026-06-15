"""
GRIT (Guide-Resolved Inference of Transcriptional Perturbation):
gRNA perturbation-driven latent transition model

Updated data design based on your prepared inputs:

Required input: a single scanpy-preprocessed h5ad file (produced by
01.mixscape.check.data.py or an equivalent pipeline) with the following
contents:

1. adata.X: cells x genes rescaled gene expression
   - shape: [N_cells, G_genes]
   - values: log-normalized + scaled (z-scored) expression, as produced by
     sc.pp.normalize_total + sc.pp.log1p + sc.pp.scale

2. adata.obsm['gRNA_counts']: cells x gRNAs sparse count matrix
   - shape: [N_cells, K_gRNAs]
   - rows: cells (aligned with adata.X)
   - column order matches adata.uns['gRNA_names']

3. adata.uns['gRNA_names']: array of gRNA name strings
   - length: K_gRNAs (matches obsm['gRNA_counts'] width)

4. gRNA -> target-gene mapping (one of):
   - Optional sidecar CSV passed via --guide-map-file with columns
     (grna, target_gene) or (guide_id, gene); non-targeting gRNAs use
     target_gene = 'NT'.
   - OR adata.obs columns 'guide_call' + 'gene' (default from
     01.mixscape.check.data.py), from which the per-gRNA target mapping is
     derived automatically.

Model design:
1. Expression encoder: rescaled expression -> reference latent state z_reference
2. Perturbation encoder:
   - gRNA count vector -> weighted gRNA embedding
   - final perturbation embedding e_g
3. FiLM fusion
4. Residual latent transition network
5. Decoder reconstructs gene expression
6. GRIT score head:
   - no external perturbation-response label CSV is required or accepted
   - exported values are model-produced GRIT scores, not observed trajectory labels
7. Cell-level output:
   - <output_prefix>_cell_level_results.tsv
   - per-cell columns: cell metadata, GRIT_score,
     plus latent vector columns (z_reference_*, z_after_perturb_*, delta_z_*)
   - GRIT_score is a target-peer-relative response strength
     (within-target shrunken z-score of ‖delta_z‖); 0 = same as peers, positive
     = stronger than peers, negative = weaker
   - z_reference is computed from the shared reference mean blended with a
     per-cell cell-state contribution, not from an observed pre-perturbation state
   - <output_prefix>_GRIT_score_metadata.tsv stores the per-target shrinkage
     statistics (within-target median, robust scale, global robust scale,
     shrinkage weight) for GRIT_score
8. Target-gene DE-like output:
   - <output_prefix>_target_gene_de_like_genes.tsv
   - compares target-gene guide pseudo-bulk replicates against reference replicates
   - <output_prefix>_target_gene_de_like_metadata.tsv stores run-level DE-like metadata
9. Perturbation-response DE output:
   - <output_prefix>_perturbation_response_de_genes.tsv
   - per (target_gene, affected_gene): cubic B-spline GAM F-test for whether the
     gene's expression varies non-trivially along GRIT_score
     (pseudotime-DE-inspired), with optional guide-level covariate adjustment;
     reports effect size, pattern (monotonic_up / monotonic_down / non_monotonic),
     peak_score, p-value, and per-target BH-FDR

Reference design:
- --reference-mode defines which cells form the shared baseline/control pool.
- The model always uses the mean expression of that reference pool as x_before.
- nt_only uses NT guide cells only.
- nt_and_unperturbed uses NT guide cells plus no-gRNA-signal cells.

Important assumption:
- You said perturbation is single perturbation, but you also prepare a gRNA count matrix.
- This implementation supports both:
    A) single dominant gRNA per cell, derived from max gRNA count
    B) weighted gRNA embedding using the full rescaled gRNA count vector
- Default below uses weighted gRNA embedding because it keeps your prepared gRNA count information.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple


def _build_arg_parser_for_early_help() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train GRIT on prepared perturb-seq matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--h5ad-file",
        type=Path,
        default=Path(r"D:\project\perturb_test\GSE213921\scanpy_perturb_obj.h5ad"),
        help=(
            "Path to the scanpy-preprocessed h5ad (X = rescaled expression, "
            "obsm['gRNA_counts'] = gRNA count matrix, uns['gRNA_names'] = "
            "gRNA names; obs['guide_call'] + obs['gene'] are used as the "
            "gRNA -> target-gene map when --guide-map-file is not provided)."
        ),
    )
    parser.add_argument(
        "--guide-map-file",
        type=Path,
        default=None,
        help=(
            "Optional CSV with columns (grna, target_gene) or (guide_id, gene). "
            "If omitted, the gRNA -> target-gene map is derived from "
            "obs['guide_call'] + obs['gene'] in the h5ad."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=script_dir / "grit",
        help=(
            "Output file prefix. Results are written to "
            "<output_prefix>_cell_level_results.tsv, "
            "<output_prefix>_GRIT_score_metadata.tsv, "
            "<output_prefix>_target_gene_de_like_genes.tsv, and "
            "<output_prefix>_perturbation_response_de_genes.tsv; "
            "DE-like run metadata is written separately."
        ),
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=script_dir / "grit_config.yaml",
        help=(
            "YAML config file with training, model, loss, QC, and DE parameters. "
            "Keys must match run_real_data_training() argument names. "
            "Missing keys fall back to function defaults."
        ),
    )

    return parser


if __name__ == "__main__" and (
    len(sys.argv) == 1 or any(arg in {"-h", "--help"} for arg in sys.argv[1:])
):
    _build_arg_parser_for_early_help().print_help()
    sys.exit(0)

import anndata
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import yaml
from patsy import dmatrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class ModelConfig:
    n_genes: int
    n_grnas: int
    n_targets: int
    latent_dim: int = 32
    grna_embed_dim: int = 32
    hidden_dim: int = 256
    dropout: float = 0.1
    use_weighted_grna_counts: bool = True
    grna_count_normalization: str = "log1p"
    top_guide_mask_k: int = 3
    guide_dropout_prob: float = 0.05
    grl_lambda: float = 0.2
    alpha_guide_residual: float = 0.1
    cell_state_dim: int = 8
    gamma_cell_state: float = 0.3
    grl_lambda_cell_state: float = 0.2


@dataclass
class LossWeights:
    rna: float = 1.0
    kl: float = 1e-3
    adv_grna_classifier: float = 0.02
    guide_residual_l2: float = 1e-3
    cell_state_kl: float = 1e-3
    cell_state_target_adversary: float = 0.02


NT_TARGET_GENE_NAME = "NT"
REFERENCE_MODE_NT_ONLY = "nt_only"
REFERENCE_MODE_NT_AND_UNPERTURBED = "nt_and_unperturbed"
DEFAULT_UNPERTURBED_REFERENCE_BINS = 20
DEFAULT_MIN_CELLS_PER_PSEUDOBULK = 3
DEFAULT_MAX_DE_PERMUTATIONS = 2000
DEFAULT_RESPONSE_SPLINE_DF = 4
DEFAULT_RESPONSE_GRID_SIZE = 100
DEFAULT_RESPONSE_MIN_CELLS_PER_TARGET = 30
DEFAULT_RESPONSE_MIN_UNIQUE_SCORES = 10
DEFAULT_RESPONSE_MIN_SHRUNKEN_SCORE_IQR = 0.05
NORMAL_IQR_TO_SD = 1.349


# ============================================================
# Data preparation utilities
# ============================================================

def _build_guide_to_target_map(
    adata: anndata.AnnData,
    grna_names: list[str],
    guide_map_file: Optional[str],
) -> Dict[str, str]:
    """
    Resolve gRNA -> target gene mapping for all gRNAs in the count matrix.

    Resolution order:
    1. Explicit CSV via guide_map_file with columns (grna, target_gene) or
       (guide_id, gene).
    2. adata.obs columns 'guide_call' + 'gene' (default from 01.mixscape pipeline).

    Guides that cannot be resolved are assigned target 'unknown' and a warning
    is printed.
    """
    guide_to_target: Dict[str, str] = {}

    if guide_map_file is not None:
        guide_map_df = pd.read_csv(guide_map_file)
        if {"grna", "target_gene"}.issubset(guide_map_df.columns):
            pass
        elif {"guide_id", "gene"}.issubset(guide_map_df.columns):
            guide_map_df = guide_map_df.rename(
                columns={"guide_id": "grna", "gene": "target_gene"}
            )
        else:
            raise ValueError(
                "guide_map CSV must contain columns (grna, target_gene) or "
                "(guide_id, gene)."
            )
        for _, row in guide_map_df.iterrows():
            guide_to_target[str(row["grna"])] = str(row["target_gene"])
    elif "guide_call" in adata.obs.columns and "gene" in adata.obs.columns:
        obs_subset = (
            adata.obs[["guide_call", "gene"]]
            .astype(str)
            .drop_duplicates()
        )
        for _, row in obs_subset.iterrows():
            guide_call = row["guide_call"]
            if guide_call in {"unassigned", "nan", "NA", ""}:
                continue
            guide_to_target[guide_call] = row["gene"]
    else:
        raise ValueError(
            "No guide-map source found. Provide --guide-map-file, or ensure the "
            "h5ad's obs contains 'guide_call' and 'gene' columns."
        )

    missing = [g for g in grna_names if g not in guide_to_target]
    if missing:
        print(
            f"Warning: {len(missing)} gRNAs have no target-gene mapping; "
            "defaulting to 'unknown'."
        )
        for g in missing:
            guide_to_target[g] = "unknown"

    return guide_to_target


def read_anndata_h5ad(
    h5ad_file: str,
    guide_map_file: Optional[str] = None,
) -> Dict[str, object]:
    """
    Read scanpy-preprocessed h5ad and prepare data for GRIT training.

    Expected h5ad structure (produced by 01.mixscape.check.data.py):
    - X: cells x genes, rescaled gene expression (log-normalized + scaled)
    - obsm["gRNA_counts"]: cells x gRNAs sparse count matrix
    - uns["gRNA_names"]: gRNA name array of length matching obsm width
    - obs["guide_call"], obs["gene"] (optional): dominant gRNA and target gene
      per cell; used to derive the gRNA -> target gene map when
      guide_map_file is not provided

    Returns dictionary with aligned tensors and mapping tables.
    """
    adata = anndata.read_h5ad(str(h5ad_file))

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    expr_array = np.asarray(X, dtype=np.float32)
    cell_names = list(map(str, adata.obs_names))
    gene_names = list(map(str, adata.var_names))

    if "gRNA_counts" not in adata.obsm:
        raise ValueError("adata.obsm['gRNA_counts'] is missing in the h5ad.")
    grna_obj = adata.obsm["gRNA_counts"]
    if hasattr(grna_obj, "toarray"):
        grna_array = grna_obj.toarray().astype(np.float32)
    else:
        grna_array = np.asarray(grna_obj, dtype=np.float32)
    if grna_array.shape[0] != expr_array.shape[0]:
        raise ValueError(
            "gRNA_counts row count does not match expression cell count "
            f"({grna_array.shape[0]} vs {expr_array.shape[0]})."
        )

    if "gRNA_names" not in adata.uns:
        raise ValueError("adata.uns['gRNA_names'] is missing in the h5ad.")
    grna_names = list(map(str, np.asarray(adata.uns["gRNA_names"]).ravel()))
    if len(grna_names) != grna_array.shape[1]:
        raise ValueError(
            f"gRNA name count ({len(grna_names)}) does not match gRNA matrix "
            f"width ({grna_array.shape[1]})."
        )

    guide_to_target = _build_guide_to_target_map(adata, grna_names, guide_map_file)
    target_gene_names_by_grna = np.array(
        [guide_to_target[g] for g in grna_names], dtype=str
    )

    dominant_grna_id_values = grna_array.argmax(axis=1).astype(np.int64)
    grna_count_sum = grna_array.sum(axis=1)
    dominant_grna_count_values = grna_array[
        np.arange(grna_array.shape[0]),
        dominant_grna_id_values,
    ]
    if grna_array.shape[1] >= 2:
        top2_counts = np.partition(grna_array, -2, axis=1)[:, -2:]
        second_grna_count_values = top2_counts[:, 0]
    else:
        second_grna_count_values = np.zeros(grna_array.shape[0], dtype=np.float32)
    dominant_grna_fraction_values = np.divide(
        dominant_grna_count_values,
        grna_count_sum,
        out=np.zeros_like(dominant_grna_count_values, dtype=np.float32),
        where=grna_count_sum > 0,
    )
    second_grna_fraction_values = np.divide(
        second_grna_count_values,
        grna_count_sum,
        out=np.zeros_like(second_grna_count_values, dtype=np.float32),
        where=grna_count_sum > 0,
    )
    dominant_target_gene_names = target_gene_names_by_grna[dominant_grna_id_values]
    unique_target_gene_names = sorted(set(target_gene_names_by_grna))
    target_gene_to_id = {
        target_gene: target_id
        for target_id, target_gene in enumerate(unique_target_gene_names)
    }
    target_gene_id_by_grna = np.array(
        [target_gene_to_id[target_gene] for target_gene in target_gene_names_by_grna],
        dtype=np.int64,
    )
    dominant_target_gene_id_values = target_gene_id_by_grna[dominant_grna_id_values]
    is_nt_control = dominant_target_gene_names == NT_TARGET_GENE_NAME

    result = {
        "x_expr": torch.tensor(expr_array),
        "c_grna": torch.tensor(grna_array, dtype=torch.float32),
        "dominant_grna_id": torch.tensor(dominant_grna_id_values, dtype=torch.long),
        "dominant_grna_count": torch.tensor(
            dominant_grna_count_values, dtype=torch.float32
        ),
        "dominant_grna_fraction": torch.tensor(
            dominant_grna_fraction_values, dtype=torch.float32
        ),
        "second_grna_count": torch.tensor(second_grna_count_values, dtype=torch.float32),
        "second_grna_fraction": torch.tensor(
            second_grna_fraction_values, dtype=torch.float32
        ),
        "dominant_target_gene_id": torch.tensor(
            dominant_target_gene_id_values, dtype=torch.long
        ),
        "is_nt_control": torch.tensor(is_nt_control, dtype=torch.bool),
        "grna_count_sum": torch.tensor(grna_count_sum, dtype=torch.float32),
        "cell_names": cell_names,
        "gene_names": gene_names,
        "grna_names": grna_names,
        "target_gene_names": list(target_gene_names_by_grna),
        "dominant_target_gene_names": list(dominant_target_gene_names),
        "target_gene_to_id": target_gene_to_id,
        "target_id_to_gene": unique_target_gene_names,
        "grna_to_target_id": torch.tensor(target_gene_id_by_grna, dtype=torch.long),
        "n_targets": len(unique_target_gene_names),
    }

    return result


# ============================================================
# Dataset
# ============================================================

class PreparedPerturbSeqDataset(Dataset):
    """
    Dataset using your prepared matrices.

    x_expr:
        [N, G] observed perturbed-after expression

    c_grna:
        [N, K] rescaled gRNA count matrix

    dominant_grna_id:
        [N] gRNA ID from max gRNA count per cell

    dominant_target_gene_id:
        [N] target-gene ID from the dominant gRNA per cell

    x_baseline_pool:
        Reference/control cells selected by --reference-mode.
        x_before is always the mean expression of this pool.
    """

    def __init__(
        self,
        x_expr: torch.Tensor,
        c_grna: torch.Tensor,
        dominant_grna_id: torch.Tensor,
        dominant_target_gene_id: torch.Tensor,
        x_baseline_pool: Optional[torch.Tensor] = None,
    ):
        if x_expr.shape[0] != c_grna.shape[0]:
            raise ValueError("x_expr and c_grna must have the same number of cells.")
        if x_expr.shape[0] != dominant_target_gene_id.shape[0]:
            raise ValueError("x_expr and dominant_target_gene_id must have the same number of cells.")
        if x_baseline_pool is None:
            raise ValueError("x_baseline_pool is required.")

        self.x_expr = x_expr.float()
        self.c_grna = c_grna.float()
        self.dominant_grna_id = dominant_grna_id.long()
        self.dominant_target_gene_id = dominant_target_gene_id.long()
        self.x_baseline_pool = x_baseline_pool.float() if x_baseline_pool is not None else None
        self.x_baseline_mean = (
            self.x_baseline_pool.mean(dim=0) if self.x_baseline_pool is not None else None
        )

    def __len__(self) -> int:
        return self.x_expr.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x_after = self.x_expr[idx]
        x_before = self.x_baseline_mean

        item = {
            "x": x_before,
            "x_before": x_before,
            "x_after": x_after,
            "c_grna": self.c_grna[idx],
            "grna_id": self.dominant_grna_id[idx],
            "target_gene_id": self.dominant_target_gene_id[idx],
        }
        return item


# ============================================================
# Neural network blocks
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


def gradient_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambda_)


# ============================================================
# Expression VAE for rescaled gene expression
# ============================================================

class ExpressionEncoder(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.backbone = MLP(n_genes, hidden_dim, hidden_dim, dropout)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10, 10)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z, mu, logvar


class ExpressionDecoder(nn.Module):
    """
    Decoder for rescaled expression.

    Since your expression matrix is already rescaled, the default reconstruction
    loss is MSE. If you later use raw UMI counts, replace this with NB/ZINB.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, n_genes: int, dropout: float):
        super().__init__()
        self.net = MLP(latent_dim, hidden_dim, n_genes, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ============================================================
# Perturbation encoder
# ============================================================

class PerturbationEncoder(nn.Module):
    """
    Uses rescaled gRNA count matrix per cell: c_grna [B, K].

    Target-shared base embedding plus a small L2-regularized guide-specific
    residual. The base captures target-level effect (shared across all guides
    that hit the same gene); the residual carries guide-specific efficiency
    or off-target variation. The residual is scaled by alpha_guide_residual
    and L2-penalized in the total loss, so it stays a controlled correction
    rather than a free guide-identity channel.

    Two modes:
    - Weighted mode:
        c_target_cell = aggregate_by_target(preprocess(c_grna))
        e_target = c_target_cell @ target_embedding_table
        e_resid = preprocess(c_grna) @ guide_residual_embedding_table
        e_grna_cell = e_target + alpha * e_resid
    - Dominant single-gRNA mode:
        e_grna_cell = target_embedding(target_id) + alpha * residual(grna_id)
    """

    def __init__(self, cfg: ModelConfig, grna_to_target_id: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.target_embedding = nn.Embedding(cfg.n_targets, cfg.grna_embed_dim)
        self.guide_residual_embedding = nn.Embedding(cfg.n_grnas, cfg.grna_embed_dim)
        nn.init.normal_(self.guide_residual_embedding.weight, mean=0.0, std=0.01)
        self.projector = MLP(cfg.grna_embed_dim, cfg.hidden_dim, cfg.grna_embed_dim, cfg.dropout)

        if grna_to_target_id.shape[0] != cfg.n_grnas:
            raise ValueError(
                "grna_to_target_id length must equal cfg.n_grnas "
                f"(got {grna_to_target_id.shape[0]} vs {cfg.n_grnas})."
            )
        self.register_buffer("grna_to_target_id", grna_to_target_id.long())

    def _top_guide_mask(self, c_grna: torch.Tensor) -> torch.Tensor:
        top_k = int(self.cfg.top_guide_mask_k)
        if top_k <= 0 or top_k >= c_grna.shape[1]:
            return c_grna

        _, top_indices = torch.topk(c_grna, k=top_k, dim=1)
        mask = torch.zeros_like(c_grna)
        mask.scatter_(1, top_indices, 1.0)
        return c_grna * mask

    def _guide_dropout(self, c_grna: torch.Tensor) -> torch.Tensor:
        dropout_prob = float(self.cfg.guide_dropout_prob)
        if not self.training or dropout_prob <= 0:
            return c_grna
        if dropout_prob >= 1:
            raise ValueError("guide_dropout_prob must be < 1.")

        active = c_grna > 0
        keep = (torch.rand_like(c_grna) >= dropout_prob) | (~active)
        dropped = c_grna * keep.to(c_grna.dtype)

        original_sum = c_grna.sum(dim=1)
        dropped_sum = dropped.sum(dim=1)
        restore_mask = (original_sum > 0) & (dropped_sum <= 0)
        if restore_mask.any():
            dominant_indices = c_grna.argmax(dim=1)
            dropped[restore_mask, dominant_indices[restore_mask]] = c_grna[
                restore_mask,
                dominant_indices[restore_mask],
            ]

        return dropped

    def _normalize_grna_counts(self, c_grna: torch.Tensor) -> torch.Tensor:
        normalization = self.cfg.grna_count_normalization
        if normalization == "none":
            return c_grna
        if normalization == "l1":
            count_sum = c_grna.sum(dim=1, keepdim=True)
            return torch.where(
                count_sum > 0,
                c_grna / count_sum.clamp_min(1e-8),
                torch.zeros_like(c_grna),
            )
        if normalization == "log1p":
            return torch.log1p(c_grna)
        raise ValueError(f"Unsupported gRNA count normalization: {normalization}")

    def preprocess_grna_counts(self, c_grna: torch.Tensor) -> torch.Tensor:
        c_grna = torch.clamp(c_grna, min=0.0)
        c_grna = self._top_guide_mask(c_grna)
        c_grna = self._guide_dropout(c_grna)
        c_grna = self._normalize_grna_counts(c_grna)
        return c_grna

    def _aggregate_to_targets(self, c_grna: torch.Tensor) -> torch.Tensor:
        # c_grna: [B, K] -> c_target: [B, n_targets]
        n_targets = int(self.cfg.n_targets)
        c_target = c_grna.new_zeros(c_grna.shape[0], n_targets)
        c_target.index_add_(1, self.grna_to_target_id, c_grna)
        return c_target

    def forward(
        self,
        c_grna: torch.Tensor,
        grna_id: torch.Tensor,
    ) -> torch.Tensor:
        alpha = float(self.cfg.alpha_guide_residual)
        if self.cfg.use_weighted_grna_counts:
            c_grna_pp = self.preprocess_grna_counts(c_grna)
            c_target = self._aggregate_to_targets(c_grna_pp)
            e_target = torch.matmul(c_target, self.target_embedding.weight)
            e_resid = torch.matmul(c_grna_pp, self.guide_residual_embedding.weight)
            e_grna = e_target + alpha * e_resid
        else:
            target_id = self.grna_to_target_id[grna_id]
            e_target = self.target_embedding(target_id)
            e_resid = self.guide_residual_embedding(grna_id)
            e_grna = e_target + alpha * e_resid

        return self.projector(e_grna)


# ============================================================
# FiLM fusion and residual transition
# ============================================================

class FiLMFusion(nn.Module):
    def __init__(self, latent_dim: int, perturb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gamma_net = MLP(perturb_dim, hidden_dim, latent_dim, dropout)
        self.beta_net = MLP(perturb_dim, hidden_dim, latent_dim, dropout)

    def forward(self, z: torch.Tensor, e_g: torch.Tensor) -> torch.Tensor:
        gamma = 1.0 + self.gamma_net(e_g)
        beta = self.beta_net(e_g)
        return gamma * z + beta


class ResidualTransitionNetwork(nn.Module):
    def __init__(self, latent_dim: int, perturb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = MLP(latent_dim + perturb_dim, hidden_dim, latent_dim, dropout)

    def forward(self, z_film: torch.Tensor, e_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_z = self.net(torch.cat([z_film, e_g], dim=-1))
        z_after_perturb = z_film + delta_z
        return z_after_perturb, delta_z


# ============================================================
# Full GRIT
# ============================================================

class GRIT(nn.Module):
    def __init__(self, cfg: ModelConfig, grna_to_target_id: torch.Tensor):
        super().__init__()
        self.cfg = cfg

        self.expression_encoder = ExpressionEncoder(
            cfg.n_genes, cfg.hidden_dim, cfg.latent_dim, cfg.dropout
        )
        self.cell_state_encoder = ExpressionEncoder(
            cfg.n_genes, cfg.hidden_dim, cfg.cell_state_dim, cfg.dropout
        )
        self.cell_state_projector = nn.Linear(cfg.cell_state_dim, cfg.latent_dim)
        self.cell_state_target_adversary = MLP(
            cfg.cell_state_dim, cfg.hidden_dim, cfg.n_targets, cfg.dropout
        )
        self.perturbation_encoder = PerturbationEncoder(cfg, grna_to_target_id)
        self.film = FiLMFusion(
            cfg.latent_dim, cfg.grna_embed_dim, cfg.hidden_dim, cfg.dropout
        )
        self.transition = ResidualTransitionNetwork(
            cfg.latent_dim, cfg.grna_embed_dim, cfg.hidden_dim, cfg.dropout
        )
        self.decoder = ExpressionDecoder(
            cfg.latent_dim, cfg.hidden_dim, cfg.n_genes, cfg.dropout
        )
        self.guide_adversary = MLP(
            cfg.latent_dim, cfg.hidden_dim, cfg.n_grnas, cfg.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        x_after: torch.Tensor,
        c_grna: torch.Tensor,
        grna_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_reference_shared, z_mu, z_logvar = self.expression_encoder(x)

        z_cell_state, cs_mu, cs_logvar = self.cell_state_encoder(x_after)
        z_reference = z_reference_shared + float(self.cfg.gamma_cell_state) * self.cell_state_projector(
            z_cell_state
        )

        e_g = self.perturbation_encoder(
            c_grna=c_grna,
            grna_id=grna_id,
        )

        z_film = self.film(z_reference, e_g)
        z_after_perturb, delta_z = self.transition(z_film, e_g)
        x_pred_after = self.decoder(z_after_perturb)
        x_delta_pred = x_pred_after - x

        delta_z_reversed = gradient_reverse(delta_z, self.cfg.grl_lambda)
        grna_logits_adv = self.guide_adversary(delta_z_reversed)

        cs_reversed = gradient_reverse(z_cell_state, self.cfg.grl_lambda_cell_state)
        cell_state_target_logits_adv = self.cell_state_target_adversary(cs_reversed)

        return {
            "z_reference_shared": z_reference_shared,
            "z_reference": z_reference,
            "z_cell_state": z_cell_state,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "cs_mu": cs_mu,
            "cs_logvar": cs_logvar,
            "e_g": e_g,
            "z_film": z_film,
            "z_after_perturb": z_after_perturb,
            "delta_z": delta_z,
            "x_recon": x_pred_after,
            "x_pred_after": x_pred_after,
            "x_delta_pred": x_delta_pred,
            "grna_logits_adv": grna_logits_adv,
            "cell_state_target_logits_adv": cell_state_target_logits_adv,
        }


# ============================================================
# Loss functions
# ============================================================

def reconstruction_mse_loss(x_target: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """
    Use this because your gene expression matrix is already rescaled.
    """
    return F.mse_loss(x_recon, x_target)


def expression_delta_mse_loss(
    x_before: torch.Tensor,
    x_after: torch.Tensor,
    x_pred_after: torch.Tensor,
) -> torch.Tensor:
    """
    Explicit residual-expression loss:
        predicted delta = predicted perturbed expression - baseline expression
        observed delta = observed perturbed expression - baseline expression

    This is algebraically equivalent to after-state MSE for a fixed baseline,
    but keeping it explicit makes the residual-transition objective auditable.
    """
    return F.mse_loss(x_pred_after - x_before, x_after - x_before)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def compute_total_loss(
    model: GRIT,
    batch: Dict[str, torch.Tensor],
    weights: LossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    x_before = batch.get("x_before", batch["x"])
    x_after = batch.get("x_after", batch["x"])

    out = model(
        x=x_before,
        x_after=x_after,
        c_grna=batch["c_grna"],
        grna_id=batch["grna_id"],
    )

    loss_rna = reconstruction_mse_loss(x_after, out["x_pred_after"])
    loss_delta = expression_delta_mse_loss(x_before, x_after, out["x_pred_after"])
    loss_kl = kl_divergence(out["z_mu"], out["z_logvar"])
    loss_adv_grna = F.cross_entropy(out["grna_logits_adv"], batch["grna_id"])
    loss_guide_residual_l2 = (
        model.perturbation_encoder.guide_residual_embedding.weight ** 2
    ).mean()
    loss_cell_state_kl = kl_divergence(out["cs_mu"], out["cs_logvar"])
    loss_cell_state_target_adv = F.cross_entropy(
        out["cell_state_target_logits_adv"],
        batch["target_gene_id"],
    )

    total = (
        weights.rna * loss_rna
        + weights.kl * loss_kl
        + weights.adv_grna_classifier * loss_adv_grna
        + weights.guide_residual_l2 * loss_guide_residual_l2
        + weights.cell_state_kl * loss_cell_state_kl
        + weights.cell_state_target_adversary * loss_cell_state_target_adv
    )

    metrics = {
        "loss_total": float(total.detach().cpu()),
        "loss_rna_mse": float(loss_rna.detach().cpu()),
        "loss_delta_mse": float(loss_delta.detach().cpu()),
        "loss_kl": float(loss_kl.detach().cpu()),
        "loss_adv_grna": float(loss_adv_grna.detach().cpu()),
        "loss_guide_residual_l2": float(loss_guide_residual_l2.detach().cpu()),
        "loss_cell_state_kl": float(loss_cell_state_kl.detach().cpu()),
        "loss_cell_state_target_adv": float(loss_cell_state_target_adv.detach().cpu()),
    }
    return total, metrics


# ============================================================
# Training
# ============================================================

def train_model(
    model: GRIT,
    train_loader: DataLoader,
    weights: LossWeights,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> GRIT:
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_metrics = []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss, metrics = compute_total_loss(model, batch, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_metrics.append(metrics)

        mean_loss = sum(m["loss_total"] for m in epoch_metrics) / max(len(epoch_metrics), 1)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | total loss = {mean_loss:.4f}")

    return model


def output_path_from_prefix(output_prefix: Path, suffix: str) -> Path:
    output_prefix = Path(output_prefix)
    return output_prefix.with_name(f"{output_prefix.name}_{suffix}")


def compute_shrunken_within_target_score(
    score: np.ndarray,
    target_gene_names: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Median-center and robust-scale target GRIT scores with target-level shrinkage.

    Targets with little within-target score spread receive a small shrinkage weight,
    so the adjusted score is pulled toward zero instead of creating an artificial
    within-target gradient.
    """
    score = np.asarray(score, dtype=np.float64)
    target_gene_names = np.asarray(target_gene_names, dtype=object).astype(str)
    adjusted = np.full(score.shape[0], np.nan, dtype=np.float64)
    median_by_cell = np.full(score.shape[0], np.nan, dtype=np.float64)
    robust_scale_by_cell = np.full(score.shape[0], np.nan, dtype=np.float64)
    shrinkage_weight_by_cell = np.full(score.shape[0], np.nan, dtype=np.float64)

    target_stats = {}
    robust_scales = []
    for target_gene in sorted(set(target_gene_names)):
        mask = (target_gene_names == target_gene) & np.isfinite(score)
        if not mask.any():
            continue

        q25, median, q75 = np.quantile(score[mask], [0.25, 0.50, 0.75])
        robust_scale = float((q75 - q25) / NORMAL_IQR_TO_SD)
        robust_scale = max(robust_scale, 0.0)
        target_stats[target_gene] = (float(median), robust_scale)
        if robust_scale > eps:
            robust_scales.append(robust_scale)

    if robust_scales:
        global_scale = float(np.median(robust_scales))
    else:
        finite_score = score[np.isfinite(score)]
        global_scale = float(np.std(finite_score)) if finite_score.size else 1.0
    if not np.isfinite(global_scale) or global_scale <= eps:
        global_scale = 1.0

    global_scale_by_cell = np.full(score.shape[0], global_scale, dtype=np.float64)
    for target_gene, (median, robust_scale) in target_stats.items():
        mask = (target_gene_names == target_gene) & np.isfinite(score)
        if not mask.any():
            continue

        if robust_scale <= eps:
            shrinkage_weight = 0.0
            adjusted_values = np.zeros(mask.sum(), dtype=np.float64)
        else:
            shrinkage_weight = (
                robust_scale**2 / (robust_scale**2 + global_scale**2)
            )
            adjusted_values = shrinkage_weight * (score[mask] - median) / robust_scale

        adjusted[mask] = adjusted_values
        median_by_cell[mask] = median
        robust_scale_by_cell[mask] = robust_scale
        shrinkage_weight_by_cell[mask] = shrinkage_weight

    return {
        "adjusted": adjusted,
        "median": median_by_cell,
        "robust_scale": robust_scale_by_cell,
        "shrinkage_weight": shrinkage_weight_by_cell,
        "global_robust_scale": global_scale_by_cell,
    }


def _build_GRIT_score_metadata_df(
    target_gene_names: np.ndarray,
    shrinkage_by_score: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    """
    Build target x score metadata long-form table from per-cell shrinkage outputs.

    Each (target_gene, score_name) row records the within-target median, robust
    scale, global robust scale, and shrinkage weight that were used to derive the
    per-cell adjusted score. These are constant within a target, so a target-level
    row is sufficient instead of repeating them per cell.
    """
    target_names_arr = np.asarray(target_gene_names, dtype=object).astype(str)
    rows = []
    for score_name in shrinkage_by_score:
        shrinkage = shrinkage_by_score[score_name]
        for target_gene in sorted(set(target_names_arr)):
            mask = target_names_arr == target_gene
            if not mask.any():
                continue
            idx = int(np.flatnonzero(mask)[0])
            rows.append(
                {
                    "target_gene": target_gene,
                    "score_name": score_name,
                    "n_cells_in_target": int(mask.sum()),
                    "within_target_median": float(shrinkage["median"][idx]),
                    "within_target_robust_scale": float(shrinkage["robust_scale"][idx]),
                    "global_robust_scale": float(shrinkage["global_robust_scale"][idx]),
                    "shrinkage_weight": float(shrinkage["shrinkage_weight"][idx]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "target_gene",
            "score_name",
            "n_cells_in_target",
            "within_target_median",
            "within_target_robust_scale",
            "global_robust_scale",
            "shrinkage_weight",
        ],
    )


def write_cell_level_results(
    model: GRIT,
    data: Dict[str, object],
    after_mask: torch.Tensor,
    baseline_pool: torch.Tensor,
    output_path: Path,
    metadata_output_path: Path,
    expression_delta_output_path: Optional[Path] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Write per-targeted-cell latent transition results for downstream R analysis.

    One within-target shrunken GRIT score is written:
    - GRIT_score: ‖delta_z‖ standardized within each target
      gene (median centering + IQR/1.349 robust scale + shrinkage). Read as a
      target-peer-relative response strength: 0 = same as target peer median,
      positive = stronger than peers, negative = weaker. Natural use:
      separating strong vs weak responders, perturbation efficacy QC, and gene
      tests along response intensity.

    Per-target shrinkage statistics (within-target median, robust scale, global
    robust scale, shrinkage weight) are written to metadata_output_path as a
    separate long-form table, one row per (target_gene, score_name) pair.

    x_before is the deterministic mean of the selected reference/control pool.
    z_reference adds a per-cell cell-state contribution on top of the encoded
    baseline mean, so it varies between cells; delta_z is z_after_perturb minus
    that per-cell z_reference.

    Without an observed perturbation-response label, the score is not calibrated
    to an external trajectory scale.
    """
    model.eval()
    model = model.to(device)

    after_indices = torch.where(after_mask)[0]
    if after_indices.numel() == 0:
        raise ValueError("No targeted perturbed cells available for cell-level output.")

    c_grna = data["c_grna"][after_indices]
    grna_id = data["dominant_grna_id"][after_indices]
    x_after_all = data["x_expr"][after_indices]
    baseline_mean = baseline_pool.float().mean(dim=0)

    z_reference_chunks = []
    z_after_perturb_chunks = []
    x_pred_after_chunks = []
    x_pred_reference_chunks = []

    with torch.no_grad():
        for start in range(0, after_indices.numel(), batch_size):
            end = min(start + batch_size, after_indices.numel())
            current_size = end - start
            x_before_batch = baseline_mean.unsqueeze(0).expand(current_size, -1).to(device)
            x_after_batch = x_after_all[start:end].to(device)
            c_grna_batch = c_grna[start:end].to(device)
            grna_id_batch = grna_id[start:end].to(device)

            out = model(
                x=x_before_batch,
                x_after=x_after_batch,
                c_grna=c_grna_batch,
                grna_id=grna_id_batch,
            )
            z_reference_chunks.append(out["z_reference"].detach().cpu())
            z_after_perturb_chunks.append(out["z_after_perturb"].detach().cpu())
            x_pred_after_chunks.append(out["x_pred_after"].detach().cpu())
            x_pred_reference_chunks.append(
                model.decoder(out["z_reference"]).detach().cpu()
            )

    z_reference = torch.cat(z_reference_chunks).numpy()
    z_after_perturb = torch.cat(z_after_perturb_chunks).numpy()
    x_pred_after = torch.cat(x_pred_after_chunks).numpy()
    x_pred_reference = torch.cat(x_pred_reference_chunks).numpy()
    pred_delta_x = x_pred_after - x_pred_reference
    delta_z = z_after_perturb - z_reference

    grna_names = data["grna_names"]
    target_gene_names = data["dominant_target_gene_names"]
    cell_names = data["cell_names"]
    selected_grna_id = grna_id.cpu().numpy()
    selected_grna_count = c_grna[
        torch.arange(c_grna.shape[0]),
        grna_id,
    ].cpu().numpy()
    selected_dominant_grna_fraction = data["dominant_grna_fraction"][after_indices].cpu().numpy()
    selected_second_grna_count = data["second_grna_count"][after_indices].cpu().numpy()
    selected_second_grna_fraction = data["second_grna_fraction"][after_indices].cpu().numpy()
    selected_target_gene_names = np.array(
        [target_gene_names[int(i)] for i in after_indices],
        dtype=object,
    )
    # Response score is the norm of the predicted perturbation effect in
    # expression (gene) space, not latent space. Latent ||delta_z|| is
    # basis-dependent and can be anti-correlated with true responder strength
    # even when the model is correct in expression space.
    pred_delta_x_norm = np.linalg.norm(pred_delta_x, axis=1)
    GRIT_score_shrinkage = compute_shrunken_within_target_score(
        score=pred_delta_x_norm,
        target_gene_names=selected_target_gene_names,
    )

    output_df = pd.DataFrame(
        {
            "cell_id": [cell_names[int(i)] for i in after_indices],
            "dominant_grna": [grna_names[int(i)] for i in selected_grna_id],
            "dominant_grna_count": selected_grna_count,
            "dominant_grna_fraction": selected_dominant_grna_fraction,
            "second_grna_count": selected_second_grna_count,
            "second_grna_fraction": selected_second_grna_fraction,
            "target_gene": selected_target_gene_names,
            "GRIT_score": GRIT_score_shrinkage["adjusted"],
        }
    )
    latent_dim = z_reference.shape[1]
    z_reference_df = pd.DataFrame(
        z_reference,
        columns=[f"z_reference_{i}" for i in range(latent_dim)],
    )
    z_after_perturb_df = pd.DataFrame(
        z_after_perturb,
        columns=[f"z_after_perturb_{i}" for i in range(latent_dim)],
    )
    delta_z_df = pd.DataFrame(
        delta_z,
        columns=[f"delta_z_{i}" for i in range(latent_dim)],
    )
    output_df = pd.concat(
        [output_df, z_reference_df, z_after_perturb_df, delta_z_df],
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, sep="\t", index=False)

    score_metadata_df = _build_GRIT_score_metadata_df(
        target_gene_names=selected_target_gene_names,
        shrinkage_by_score={
            "GRIT_score": GRIT_score_shrinkage,
        },
    )
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    score_metadata_df.to_csv(metadata_output_path, sep="\t", index=False)

    if expression_delta_output_path is not None:
        gene_names = list(data["gene_names"])
        pred_delta_x_df = pd.DataFrame(pred_delta_x, columns=gene_names)
        pred_delta_x_df.insert(
            0, "cell_id", [cell_names[int(i)] for i in after_indices]
        )
        expression_delta_output_path.parent.mkdir(parents=True, exist_ok=True)
        pred_delta_x_df.to_csv(
            expression_delta_output_path,
            sep="\t",
            index=False,
            float_format="%.6g",
        )

    return output_df


def build_reference_masks(
    data: Dict[str, object],
    is_unperturbed: torch.Tensor,
    reference_mode: str,
    guide_qc_pass: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build reference masks shared by model baseline construction and DE-like analysis.

    NT reference cells require an NT dominant guide and nonzero gRNA signal.
    Unperturbed reference cells are defined by total gRNA count threshold upstream.
    """
    if reference_mode not in {REFERENCE_MODE_NT_ONLY, REFERENCE_MODE_NT_AND_UNPERTURBED}:
        raise ValueError(
            "reference_mode must be one of: "
            f"{REFERENCE_MODE_NT_ONLY}, {REFERENCE_MODE_NT_AND_UNPERTURBED}"
        )

    if guide_qc_pass is None:
        guide_qc_pass = torch.ones_like(is_unperturbed, dtype=torch.bool)

    nt_reference_mask = data["is_nt_control"] & (~is_unperturbed) & guide_qc_pass
    unperturbed_reference_mask = torch.zeros_like(is_unperturbed, dtype=torch.bool)
    if reference_mode == REFERENCE_MODE_NT_AND_UNPERTURBED:
        unperturbed_reference_mask = is_unperturbed

    reference_mask = nt_reference_mask | unperturbed_reference_mask
    if int(reference_mask.sum().item()) == 0:
        raise ValueError(f"reference_mode={reference_mode!r} produced zero reference cells.")

    return reference_mask, nt_reference_mask, unperturbed_reference_mask


def build_multi_guide_qc_mask(
    data: Dict[str, object],
    is_unperturbed: torch.Tensor,
    enabled: bool = True,
    min_dominant_guide_fraction: float = 0.5,
    max_second_guide_fraction: float = 0.3,
) -> torch.Tensor:
    """
    Flag clean single-guide cells while preserving no-gRNA unperturbed cells.
    """
    if not enabled:
        return torch.ones_like(is_unperturbed, dtype=torch.bool)

    dominant_fraction = data["dominant_grna_fraction"]
    second_fraction = data["second_grna_fraction"]
    clean_guided = (
        (dominant_fraction >= min_dominant_guide_fraction)
        & (second_fraction <= max_second_guide_fraction)
    )
    return is_unperturbed | clean_guided


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full_like(p_values, np.nan, dtype=np.float64)
    valid = np.isfinite(p_values)
    if not valid.any():
        return adjusted

    valid_indices = np.where(valid)[0]
    valid_p = p_values[valid]
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    n_tests = ranked_p.size
    bh = ranked_p * n_tests / np.arange(1, n_tests + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    adjusted[valid_indices[order]] = np.minimum(bh, 1.0)
    return adjusted


def _welch_t_statistic(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    n_a = group_a.shape[0]
    n_b = group_b.shape[0]
    if n_a < 2 or n_b < 2:
        return np.full(group_a.shape[1], np.nan, dtype=np.float64)

    mean_a = group_a.mean(axis=0)
    mean_b = group_b.mean(axis=0)
    var_a = group_a.var(axis=0, ddof=1)
    var_b = group_b.var(axis=0, ddof=1)
    denom = np.sqrt((var_a / n_a) + (var_b / n_b))
    diff = mean_a - mean_b
    stat = np.divide(
        diff,
        denom,
        out=np.zeros_like(diff, dtype=np.float64),
        where=denom > 0,
    )
    stat[(denom == 0) & (diff > 0)] = np.inf
    stat[(denom == 0) & (diff < 0)] = -np.inf
    return stat


def _permutation_p_values_for_welch_t(
    group_a: np.ndarray,
    group_b: np.ndarray,
    observed_stat: np.ndarray,
    rng: np.random.Generator,
    max_permutations: int = DEFAULT_MAX_DE_PERMUTATIONS,
) -> np.ndarray:
    n_a = group_a.shape[0]
    n_b = group_b.shape[0]
    n_total = n_a + n_b
    if n_a < 2 or n_b < 2:
        return np.full(group_a.shape[1], np.nan, dtype=np.float64)

    combined = np.vstack([group_a, group_b])
    observed_abs = np.abs(observed_stat)
    valid = np.isfinite(observed_abs)
    exceed_count = np.zeros(group_a.shape[1], dtype=np.float64)
    n_permutations = 0

    n_combinations = math.comb(n_total, n_a)
    if n_combinations <= max_permutations:
        combo_iter = itertools.combinations(range(n_total), n_a)
    else:
        combo_iter = (
            tuple(sorted(rng.choice(n_total, size=n_a, replace=False).tolist()))
            for _ in range(max_permutations)
        )

    for combo in combo_iter:
        group_a_mask = np.zeros(n_total, dtype=bool)
        group_a_mask[list(combo)] = True
        perm_stat = _welch_t_statistic(combined[group_a_mask], combined[~group_a_mask])
        exceed_count += (np.abs(perm_stat) >= observed_abs) & valid
        n_permutations += 1

    p_values = (exceed_count + 1.0) / (n_permutations + 1.0)
    p_values[~valid] = np.nan
    return p_values


def _guide_pseudobulk_replicates(
    x_expr: np.ndarray,
    dominant_grna_id: np.ndarray,
    cell_mask: np.ndarray,
    grna_names: list[str],
    min_cells_per_guide: int = DEFAULT_MIN_CELLS_PER_PSEUDOBULK,
) -> Tuple[np.ndarray, list[str], list[int], list[int]]:
    guide_ids = sorted(np.unique(dominant_grna_id[cell_mask]).tolist())
    replicates = []
    replicate_names = []
    replicate_ids = []
    cell_counts = []

    for guide_id in guide_ids:
        guide_mask = cell_mask & (dominant_grna_id == guide_id)
        n_cells = int(guide_mask.sum())
        if n_cells < min_cells_per_guide:
            continue
        replicates.append(x_expr[guide_mask].mean(axis=0))
        replicate_names.append(grna_names[int(guide_id)])
        replicate_ids.append(int(guide_id))
        cell_counts.append(n_cells)

    if not replicates:
        return (
            np.empty((0, x_expr.shape[1]), dtype=np.float64),
            [],
            [],
            [],
        )
    return np.vstack(replicates), replicate_names, replicate_ids, cell_counts


def _unperturbed_bin_pseudobulk_replicates(
    x_expr: np.ndarray,
    cell_mask: np.ndarray,
    rng: np.random.Generator,
    n_bins: int = DEFAULT_UNPERTURBED_REFERENCE_BINS,
) -> np.ndarray:
    cell_indices = np.flatnonzero(cell_mask)
    if cell_indices.size == 0:
        return np.empty((0, x_expr.shape[1]), dtype=np.float64)

    shuffled = cell_indices.copy()
    rng.shuffle(shuffled)
    n_bins = max(1, min(n_bins, shuffled.size))
    bins = np.array_split(shuffled, n_bins)
    return np.vstack([x_expr[bin_indices].mean(axis=0) for bin_indices in bins])


def write_target_gene_de_like_genes(
    data: Dict[str, object],
    after_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    nt_reference_mask: torch.Tensor,
    unperturbed_reference_mask: torch.Tensor,
    reference_mode: str,
    output_path: Path,
    metadata_output_path: Optional[Path] = None,
    seed: int = 7,
    min_cells_per_guide: int = DEFAULT_MIN_CELLS_PER_PSEUDOBULK,
    unperturbed_reference_bins: int = DEFAULT_UNPERTURBED_REFERENCE_BINS,
    max_permutations: int = DEFAULT_MAX_DE_PERMUTATIONS,
) -> None:
    """
    Write target-gene DE-like genes using guide-level pseudo-bulk replicates.

    For each target gene and each expression gene, target-gene guide means are
    compared with reference pseudo-replicates. NT reference cells are summarized
    per NT guide; unperturbed reference cells are summarized into deterministic
    random bins when reference_mode='nt_and_unperturbed'.
    """
    rng = np.random.default_rng(seed)
    x_expr = data["x_expr"].cpu().numpy().astype(np.float64)
    dominant_grna_id = data["dominant_grna_id"].cpu().numpy()
    after_mask_np = after_mask.cpu().numpy().astype(bool)
    reference_mask_np = reference_mask.cpu().numpy().astype(bool)
    nt_reference_mask_np = nt_reference_mask.cpu().numpy().astype(bool)
    unperturbed_reference_mask_np = unperturbed_reference_mask.cpu().numpy().astype(bool)

    gene_names = list(data["gene_names"])
    grna_names = list(data["grna_names"])
    dominant_target_gene_names = np.array(data["dominant_target_gene_names"], dtype=object)

    nt_reference_reps, control_guides, _, _ = _guide_pseudobulk_replicates(
        x_expr=x_expr,
        dominant_grna_id=dominant_grna_id,
        cell_mask=nt_reference_mask_np,
        grna_names=grna_names,
        min_cells_per_guide=min_cells_per_guide,
    )
    unperturbed_reps = np.empty((0, x_expr.shape[1]), dtype=np.float64)
    if reference_mode == REFERENCE_MODE_NT_AND_UNPERTURBED:
        unperturbed_reps = _unperturbed_bin_pseudobulk_replicates(
            x_expr=x_expr,
            cell_mask=unperturbed_reference_mask_np,
            rng=rng,
            n_bins=unperturbed_reference_bins,
        )

    reference_parts = []
    if nt_reference_reps.shape[0] > 0:
        reference_parts.append(nt_reference_reps)
    if unperturbed_reps.shape[0] > 0:
        reference_parts.append(unperturbed_reps)
    if not reference_parts:
        raise ValueError("No valid reference pseudo-bulk replicates were available.")

    reference_reps = np.vstack(reference_parts)
    if reference_reps.shape[0] < 2:
        raise ValueError(
            "At least two reference pseudo-bulk replicates are required for DE-like testing."
        )

    target_genes = sorted(
        {
            str(target)
            for target in dominant_target_gene_names[after_mask_np]
            if str(target) != NT_TARGET_GENE_NAME
        }
    )
    if not target_genes:
        raise ValueError("No targeted perturbed cells are available for DE-like testing.")

    all_target_dfs = []
    control_guides_text = ";".join(control_guides)
    n_cells_reference = int(reference_mask_np.sum())
    n_cells_nt_reference = int(nt_reference_mask_np.sum())
    n_cells_unperturbed_reference = int(unperturbed_reference_mask_np.sum())
    n_nt_control_guides = len(control_guides)
    n_unperturbed_bins = int(unperturbed_reps.shape[0])
    n_control_replicates = int(reference_reps.shape[0])

    for target_gene in target_genes:
        target_cell_mask = after_mask_np & (dominant_target_gene_names == target_gene)
        perturbed_reps, guides, guide_ids, _ = _guide_pseudobulk_replicates(
            x_expr=x_expr,
            dominant_grna_id=dominant_grna_id,
            cell_mask=target_cell_mask,
            grna_names=grna_names,
            min_cells_per_guide=min_cells_per_guide,
        )
        if perturbed_reps.shape[0] < 2:
            print(
                f"Skipping target_gene={target_gene}: "
                f"valid guide pseudo-bulk replicates={perturbed_reps.shape[0]} < 2"
            )
            continue

        used_target_cell_mask = target_cell_mask & np.isin(dominant_grna_id, guide_ids)
        n_cells_perturbed = int(used_target_cell_mask.sum())
        n_guides = int(perturbed_reps.shape[0])

        mean_expr_perturbed = perturbed_reps.mean(axis=0)
        mean_expr_reference = reference_reps.mean(axis=0)
        mean_expr_diff = mean_expr_perturbed - mean_expr_reference
        median_expr_perturbed = np.median(perturbed_reps, axis=0)
        median_expr_reference = np.median(reference_reps, axis=0)
        frac_expr_perturbed = (x_expr[used_target_cell_mask] > 0).mean(axis=0)
        frac_expr_reference = (x_expr[reference_mask_np] > 0).mean(axis=0)
        frac_expr_diff = frac_expr_perturbed - frac_expr_reference

        welch_t_statistic = _welch_t_statistic(perturbed_reps, reference_reps)
        p_value = _permutation_p_values_for_welch_t(
            group_a=perturbed_reps,
            group_b=reference_reps,
            observed_stat=welch_t_statistic,
            rng=rng,
            max_permutations=max_permutations,
        )
        fdr = _benjamini_hochberg(p_value)

        guide_diffs = perturbed_reps - mean_expr_reference[None, :]
        direction = np.where(
            mean_expr_diff > 0,
            "up",
            np.where(mean_expr_diff < 0, "down", "ns"),
        )
        n_guides_supporting_direction = np.where(
            mean_expr_diff > 0,
            (guide_diffs > 0).sum(axis=0),
            np.where(mean_expr_diff < 0, (guide_diffs < 0).sum(axis=0), 0),
        ).astype(int)
        frac_guides_supporting_direction = n_guides_supporting_direction / n_guides

        target_df = pd.DataFrame(
            {
                "target_gene": target_gene,
                "affected_gene": gene_names,
                "n_guides": n_guides,
                "n_cells_perturbed": n_cells_perturbed,
                "guides": ";".join(guides),
                "mean_expr_perturbed": mean_expr_perturbed,
                "mean_expr_reference": mean_expr_reference,
                "mean_expr_diff": mean_expr_diff,
                "median_expr_perturbed": median_expr_perturbed,
                "median_expr_reference": median_expr_reference,
                "frac_expr_perturbed": frac_expr_perturbed,
                "frac_expr_reference": frac_expr_reference,
                "frac_expr_diff": frac_expr_diff,
                "welch_t_statistic": welch_t_statistic,
                "p_value": p_value,
                "fdr": fdr,
                "direction": direction,
                "n_guides_supporting_direction": n_guides_supporting_direction,
                "frac_guides_supporting_direction": frac_guides_supporting_direction,
            }
        )
        target_df["_fdr_sort"] = target_df["fdr"].fillna(1.0)
        target_df["_mean_diff_sort"] = target_df["mean_expr_diff"].abs().fillna(0.0)
        target_df = target_df.sort_values(
            ["_fdr_sort", "_mean_diff_sort"],
            ascending=[True, False],
        ).reset_index(drop=True)
        target_df = target_df.drop(
            columns=["_fdr_sort", "_mean_diff_sort"]
        )
        final_columns = [
            "target_gene",
            "affected_gene",
            "n_guides",
            "n_cells_perturbed",
            "guides",
            "mean_expr_perturbed",
            "mean_expr_reference",
            "mean_expr_diff",
            "median_expr_perturbed",
            "median_expr_reference",
            "frac_expr_perturbed",
            "frac_expr_reference",
            "frac_expr_diff",
            "welch_t_statistic",
            "p_value",
            "fdr",
            "direction",
            "n_guides_supporting_direction",
            "frac_guides_supporting_direction",
        ]
        all_target_dfs.append(target_df[final_columns])

    if not all_target_dfs:
        raise ValueError("No target genes had enough guide pseudo-bulk replicates for testing.")

    output_df = pd.concat(all_target_dfs, axis=0, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, sep="\t", index=False)

    if metadata_output_path is not None:
        metadata_records = [
            ("analysis", "target_gene_de_like_genes"),
            ("output_file", str(output_path)),
            ("reference_mode", reference_mode),
            ("test_method", "welch_t_permutation_on_guide_pseudobulk"),
            ("n_control_replicates", n_control_replicates),
            ("n_nt_control_guides", n_nt_control_guides),
            ("n_unperturbed_bins", n_unperturbed_bins),
            ("n_cells_reference", n_cells_reference),
            ("n_cells_nt_reference", n_cells_nt_reference),
            ("n_cells_unperturbed_reference", n_cells_unperturbed_reference),
            ("control_guides", control_guides_text),
            ("min_cells_per_guide", min_cells_per_guide),
            ("unperturbed_reference_bins", unperturbed_reference_bins),
            ("max_permutations", max_permutations),
            ("n_target_genes_tested", len(all_target_dfs)),
            ("n_result_rows", output_df.shape[0]),
        ]
        metadata_df = pd.DataFrame(metadata_records, columns=["key", "value"])
        metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(metadata_output_path, sep="\t", index=False)


_DE_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "target_gene",
    "affected_gene",
    "n_cells",
    "n_guides",
    "spline_df",
    "endpoint_diff",
    "curve_range_signed",
    "mean_expr_low_score",
    "mean_expr_high_score",
    "f_statistic",
    "p_value",
    "fdr",
    "pattern",
    "peak_score",
)


def _spline_de_one_target(
    target_gene: str,
    score: np.ndarray,
    expression: np.ndarray,
    guide_ids: Optional[np.ndarray],
    gene_names: list[str],
    spline_df: int,
    grid_size: int,
) -> Optional[pd.DataFrame]:
    """
    Fit a cubic B-spline GAM for one target across all genes.

    Model:
        alt:  expression ~ intercept + bs(score, df=spline_df) [+ guide_dummies]
        null: expression ~ intercept                          [+ guide_dummies]
    F-test compares the two via residual sum of squares. Guide dummies are
    included if guide_ids is given and contains >= 2 unique values, controlling
    for guide-level confounding so the spline term captures the score-driven
    component.

    Returns per-(target, gene) DataFrame, or None if degrees of freedom are
    insufficient for the spline fit.
    """
    n_cells = int(score.shape[0])
    n_genes = int(expression.shape[1])

    spline_design = dmatrix(
        f"bs(x, df={int(spline_df)}, degree=3, include_intercept=False)",
        {"x": score},
        return_type="dataframe",
    )
    spline_basis = np.asarray(spline_design, dtype=np.float64)
    spline_design_info = spline_design.design_info
    intercept = np.ones((n_cells, 1), dtype=np.float64)

    guide_dummies: Optional[np.ndarray] = None
    if guide_ids is not None and np.unique(guide_ids).size > 1:
        guide_dummies = pd.get_dummies(
            guide_ids, drop_first=True
        ).to_numpy(dtype=np.float64)

    if guide_dummies is None:
        X_alt = np.hstack([intercept, spline_basis])
        X_null = intercept
    else:
        X_alt = np.hstack([intercept, spline_basis, guide_dummies])
        X_null = np.hstack([intercept, guide_dummies])

    df_diff = int(X_alt.shape[1] - X_null.shape[1])
    df_resid = int(n_cells - X_alt.shape[1])
    if df_resid <= 0 or df_diff <= 0:
        return None

    beta_alt, _, _, _ = np.linalg.lstsq(X_alt, expression, rcond=None)
    beta_null, _, _, _ = np.linalg.lstsq(X_null, expression, rcond=None)
    resid_alt = expression - X_alt @ beta_alt
    resid_null = expression - X_null @ beta_null
    rss_alt = (resid_alt ** 2).sum(axis=0)
    rss_null = (resid_null ** 2).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_stat = ((rss_null - rss_alt) / df_diff) / (rss_alt / df_resid)
    p_value = np.full(n_genes, np.nan, dtype=np.float64)
    finite_f = np.isfinite(f_stat) & (f_stat >= 0)
    if finite_f.any():
        p_value[finite_f] = scipy.stats.f.sf(f_stat[finite_f], df_diff, df_resid)
    fdr = _benjamini_hochberg(p_value)

    score_min = float(score.min())
    score_max = float(score.max())
    score_grid = np.linspace(score_min, score_max, int(grid_size))
    grid_basis = np.asarray(
        dmatrix(spline_design_info, {"x": score_grid}, return_type="dataframe"),
        dtype=np.float64,
    )
    grid_intercept = np.ones((int(grid_size), 1), dtype=np.float64)
    if guide_dummies is not None:
        n_guide_dummies = guide_dummies.shape[1]
        grid_guide_zeros = np.zeros((int(grid_size), n_guide_dummies), dtype=np.float64)
        X_grid = np.hstack([grid_intercept, grid_basis, grid_guide_zeros])
    else:
        X_grid = np.hstack([grid_intercept, grid_basis])
    fitted_curves = X_grid @ beta_alt  # [grid_size, n_genes]

    curve_max = fitted_curves.max(axis=0)
    curve_min = fitted_curves.min(axis=0)
    peak_score_per_gene = score_grid[np.argmax(fitted_curves, axis=0)]

    score_q10 = float(np.quantile(score, 0.10))
    score_q90 = float(np.quantile(score, 0.90))
    q10_idx = int(np.argmin(np.abs(score_grid - score_q10)))
    q90_idx = int(np.argmin(np.abs(score_grid - score_q90)))
    endpoint_diff = fitted_curves[q90_idx] - fitted_curves[q10_idx]
    endpoint_sign = np.where(endpoint_diff >= 0, 1.0, -1.0)
    curve_range_signed = (curve_max - curve_min) * endpoint_sign

    derivatives = np.diff(fitted_curves, axis=0)
    n_steps = derivatives.shape[0]
    n_up = (derivatives > 0).sum(axis=0)
    n_down = (derivatives < 0).sum(axis=0)
    pattern = np.where(
        n_up >= 0.9 * n_steps,
        "monotonic_up",
        np.where(n_down >= 0.9 * n_steps, "monotonic_down", "non_monotonic"),
    )

    score_low_threshold = float(np.quantile(score, 0.25))
    score_high_threshold = float(np.quantile(score, 0.75))
    low_mask = score <= score_low_threshold
    high_mask = score >= score_high_threshold
    mean_expr_low = (
        expression[low_mask].mean(axis=0)
        if low_mask.any()
        else np.full(n_genes, np.nan, dtype=np.float64)
    )
    mean_expr_high = (
        expression[high_mask].mean(axis=0)
        if high_mask.any()
        else np.full(n_genes, np.nan, dtype=np.float64)
    )

    n_guides = int(np.unique(guide_ids).size) if guide_ids is not None else 1

    target_df_out = pd.DataFrame(
        {
            "target_gene": str(target_gene),
            "affected_gene": gene_names,
            "n_cells": n_cells,
            "n_guides": n_guides,
            "spline_df": int(spline_df),
            "endpoint_diff": endpoint_diff,
            "curve_range_signed": curve_range_signed,
            "mean_expr_low_score": mean_expr_low,
            "mean_expr_high_score": mean_expr_high,
            "f_statistic": f_stat,
            "p_value": p_value,
            "fdr": fdr,
            "pattern": pattern,
            "peak_score": peak_score_per_gene,
        }
    )
    target_df_out["_fdr_sort"] = target_df_out["fdr"].fillna(1.0)
    target_df_out["_eff_sort"] = target_df_out["curve_range_signed"].abs().fillna(0.0)
    target_df_out = (
        target_df_out.sort_values(
            ["_fdr_sort", "_eff_sort"], ascending=[True, False]
        )
        .drop(columns=["_fdr_sort", "_eff_sort"])
        .reset_index(drop=True)
    )
    return target_df_out


def write_perturbation_response_de_genes(
    cell_level_df: pd.DataFrame,
    expression_values: np.ndarray,
    gene_names: list[str],
    output_path: Path,
    score_name: str = "GRIT_score",
    spline_df: int = DEFAULT_RESPONSE_SPLINE_DF,
    include_guide_covariate: bool = True,
    min_cells_per_target: int = DEFAULT_RESPONSE_MIN_CELLS_PER_TARGET,
    min_unique_scores: int = DEFAULT_RESPONSE_MIN_UNIQUE_SCORES,
    min_shrunken_score_iqr: float = DEFAULT_RESPONSE_MIN_SHRUNKEN_SCORE_IQR,
    grid_size: int = DEFAULT_RESPONSE_GRID_SIZE,
) -> None:
    """
    Per-target DE test along GRIT_score using a cubic B-spline GAM.

    Pseudotime-DE-inspired test: for each target gene's cells, fit
        expression ~ bs(score, df=spline_df, degree=3) + C(dominant_grna)
    versus null
        expression ~ C(dominant_grna)
    via OLS, compared with an F-test. dominant_grna inclusion controls for
    guide-level confounding so the spline term captures only the score-driven
    component (set include_guide_covariate=False to drop it).

    Effect size columns:
      - endpoint_diff: fitted(Q90 of score) minus fitted(Q10 of score)
      - curve_range_signed: (curve_max - curve_min) signed by endpoint direction
    pattern: monotonic_up / monotonic_down / non_monotonic from derivative signs.
    peak_score: score location of the fitted curve maximum, useful for
    non_monotonic genes.

    P-values are analytical (F survival function); per-target BH-FDR is reported.
    """
    if expression_values.shape[0] != cell_level_df.shape[0]:
        raise ValueError("expression_values and cell_level_df must have the same number of cells.")

    cell_level_df = cell_level_df.reset_index(drop=True)
    expression_values = np.asarray(expression_values, dtype=np.float64)
    gene_names = list(gene_names)
    result_tables: list[pd.DataFrame] = []

    for target_gene, target_df in cell_level_df.groupby("target_gene", sort=True):
        target_indices_full = target_df.index.to_numpy()
        if int(target_indices_full.size) < min_cells_per_target:
            print(
                f"Skipping response DE target_gene={target_gene}: "
                f"n_cells={target_indices_full.size} < {min_cells_per_target}"
            )
            continue

        if score_name not in target_df.columns:
            continue

        score_full = pd.to_numeric(target_df[score_name], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite_mask = np.isfinite(score_full)
        score = score_full[finite_mask]
        target_indices = target_indices_full[finite_mask]
        n_cells = int(score.shape[0])
        if n_cells < min_cells_per_target:
            continue

        unique_scores = np.unique(score)
        if unique_scores.size < min_unique_scores:
            print(
                f"Skipping response DE target_gene={target_gene}: "
                f"unique_scores={unique_scores.size} < {min_unique_scores}"
            )
            continue

        score_q25, _, score_q75 = np.quantile(score, [0.25, 0.50, 0.75])
        score_iqr = float(score_q75 - score_q25)
        if score_iqr <= 0 or not np.isfinite(score_iqr) or score_iqr < min_shrunken_score_iqr:
            print(
                f"Skipping response DE target_gene={target_gene}: "
                f"shrunken score_iqr={score_iqr:.4g} < {min_shrunken_score_iqr}"
            )
            continue

        target_expression = expression_values[target_indices]
        guide_ids: Optional[np.ndarray] = None
        if include_guide_covariate:
            guide_ids = (
                target_df["dominant_grna"].to_numpy()[finite_mask].astype(str)
            )

        target_df_out = _spline_de_one_target(
            target_gene=str(target_gene),
            score=score,
            expression=target_expression,
            guide_ids=guide_ids,
            gene_names=gene_names,
            spline_df=int(spline_df),
            grid_size=int(grid_size),
        )
        if target_df_out is None:
            print(
                f"Skipping response DE target_gene={target_gene}: "
                "insufficient residual df for spline fit"
            )
            continue
        result_tables.append(target_df_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if result_tables:
        output_df = pd.concat(result_tables, axis=0, ignore_index=True)
    else:
        output_df = pd.DataFrame(columns=list(_DE_OUTPUT_COLUMNS))
    output_df.to_csv(output_path, sep="\t", index=False)


# ============================================================
# Real-data run parameters
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train GRIT on prepared perturb-seq matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--h5ad-file",
        type=Path,
        default=Path(r"D:\project\perturb_test\GSE213921\scanpy_perturb_obj.h5ad"),
        help=(
            "Path to the scanpy-preprocessed h5ad (X = rescaled expression, "
            "obsm['gRNA_counts'] = gRNA count matrix, uns['gRNA_names'] = "
            "gRNA names; obs['guide_call'] + obs['gene'] are used as the "
            "gRNA -> target-gene map when --guide-map-file is not provided)."
        ),
    )
    parser.add_argument(
        "--guide-map-file",
        type=Path,
        default=None,
        help=(
            "Optional CSV with columns (grna, target_gene) or (guide_id, gene). "
            "If omitted, the gRNA -> target-gene map is derived from "
            "obs['guide_call'] + obs['gene'] in the h5ad."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=SCRIPT_DIR / "grit",
        help=(
            "Output file prefix. Results are written to "
            "<output_prefix>_cell_level_results.tsv, "
            "<output_prefix>_GRIT_score_metadata.tsv, "
            "<output_prefix>_target_gene_de_like_genes.tsv, and "
            "<output_prefix>_perturbation_response_de_genes.tsv; "
            "DE-like run metadata is written separately."
        ),
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=SCRIPT_DIR / "grit_config.yaml",
        help=(
            "YAML config file with training, model, loss, QC, and DE parameters. "
            "Keys must match run_real_data_training() argument names. "
            "Missing keys fall back to function defaults."
        ),
    )

    return parser


def run_real_data_training(
    h5ad_file: Path = Path(r"D:\project\perturb_test\GSE213921\scanpy_perturb_obj.h5ad"),
    guide_map_file: Optional[Path] = None,
    seed: int = 7,
    batch_size: int = 64,
    n_epochs: int = 20,
    learning_rate: float = 1e-3,
    output_prefix: Path = SCRIPT_DIR / "grit",
    reference_mode: str = REFERENCE_MODE_NT_AND_UNPERTURBED,
    unperturbed_grna_threshold: float = 0.0,
    latent_dim: int = 32,
    grna_embed_dim: int = 32,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    use_weighted_grna_counts: bool = True,
    grna_count_normalization: str = "log1p",
    top_guide_mask_k: int = 3,
    guide_dropout_prob: float = 0.05,
    multi_guide_qc: bool = True,
    min_dominant_guide_fraction: float = 0.5,
    max_second_guide_fraction: float = 0.3,
    grl_lambda: float = 0.2,
    alpha_guide_residual: float = 0.1,
    cell_state_dim: int = 8,
    gamma_cell_state: float = 0.3,
    grl_lambda_cell_state: float = 0.2,
    loss_rna: float = 1.0,
    loss_kl: float = 1e-3,
    loss_adv_grna_classifier: float = 0.02,
    loss_guide_residual_l2: float = 1e-3,
    loss_cell_state_kl: float = 1e-3,
    loss_cell_state_target_adversary: float = 0.02,
    response_spline_df: int = DEFAULT_RESPONSE_SPLINE_DF,
    response_include_guide_covariate: bool = True,
    response_min_cells_per_target: int = DEFAULT_RESPONSE_MIN_CELLS_PER_TARGET,
    response_min_unique_scores: int = DEFAULT_RESPONSE_MIN_UNIQUE_SCORES,
    response_min_shrunken_score_iqr: float = DEFAULT_RESPONSE_MIN_SHRUNKEN_SCORE_IQR,
) -> GRIT:
    """
    Run real-data training.

    The default parameter values mirror the previous inline example settings.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if response_spline_df < 2:
        raise ValueError("--response-spline-df must be >= 2.")
    if response_min_cells_per_target < 2:
        raise ValueError("--response-min-cells-per-target must be >= 2.")
    if response_min_unique_scores < 2:
        raise ValueError("--response-min-unique-scores must be >= 2.")
    if response_min_shrunken_score_iqr < 0:
        raise ValueError("--response-min-shrunken-score-iqr must be >= 0.")
    if grna_count_normalization not in {"none", "l1", "log1p"}:
        raise ValueError("--grna-count-normalization must be one of: none, l1, log1p.")
    if top_guide_mask_k < 0:
        raise ValueError("--top-guide-mask-k must be >= 0.")
    if not 0 <= guide_dropout_prob < 1:
        raise ValueError("--guide-dropout-prob must be in the interval [0, 1).")
    if alpha_guide_residual < 0:
        raise ValueError("--alpha-guide-residual must be >= 0.")
    if loss_guide_residual_l2 < 0:
        raise ValueError("--loss-guide-residual-l2 must be >= 0.")
    if cell_state_dim < 1:
        raise ValueError("--cell-state-dim must be >= 1.")
    if gamma_cell_state < 0:
        raise ValueError("--gamma-cell-state must be >= 0.")
    if grl_lambda_cell_state < 0:
        raise ValueError("--grl-lambda-cell-state must be >= 0.")
    if loss_cell_state_kl < 0:
        raise ValueError("--loss-cell-state-kl must be >= 0.")
    if loss_cell_state_target_adversary < 0:
        raise ValueError("--loss-cell-state-target-adversary must be >= 0.")
    if not 0 <= min_dominant_guide_fraction <= 1:
        raise ValueError("--min-dominant-guide-fraction must be in [0, 1].")
    if not 0 <= max_second_guide_fraction <= 1:
        raise ValueError("--max-second-guide-fraction must be in [0, 1].")

    h5ad_file = Path(h5ad_file)
    data = read_anndata_h5ad(
        h5ad_file=str(h5ad_file),
        guide_map_file=str(guide_map_file) if guide_map_file is not None else None,
    )

    n_cells = data["x_expr"].shape[0]
    n_genes = data["x_expr"].shape[1]
    n_grnas = data["c_grna"].shape[1]
    is_unperturbed = data["grna_count_sum"] <= unperturbed_grna_threshold
    guide_qc_pass = build_multi_guide_qc_mask(
        data=data,
        is_unperturbed=is_unperturbed,
        enabled=multi_guide_qc,
        min_dominant_guide_fraction=min_dominant_guide_fraction,
        max_second_guide_fraction=max_second_guide_fraction,
    )
    reference_mask, nt_reference_mask, unperturbed_reference_mask = build_reference_masks(
        data=data,
        is_unperturbed=is_unperturbed,
        reference_mode=reference_mode,
        guide_qc_pass=guide_qc_pass,
    )
    n_nt_control_cells = int(nt_reference_mask.sum().item())
    n_unperturbed_cells = int(is_unperturbed.sum().item())
    n_baseline_cells = int(reference_mask.sum().item())
    n_multi_guide_qc_failed = int((~guide_qc_pass & ~is_unperturbed).sum().item())
    if n_baseline_cells == 0:
        raise ValueError(f"reference_mode={reference_mode!r} produced zero reference cells.")
    baseline_pool = data["x_expr"][reference_mask]

    loaded_parts = [
        "Loaded data:",
        f"cells={n_cells}",
        f"genes={n_genes}",
        f"gRNAs={n_grnas}",
        f"reference_mode={reference_mode}",
        "baseline=reference_mean",
        f"NT_controls={n_nt_control_cells}",
        f"not_perturbed={n_unperturbed_cells}",
        f"multi_guide_qc={multi_guide_qc}",
        f"multi_guide_qc_failed={n_multi_guide_qc_failed}",
    ]
    loaded_parts.append(f"baseline_cells={n_baseline_cells}")
    print(*loaded_parts)

    after_mask = (~data["is_nt_control"]) & (~is_unperturbed) & guide_qc_pass
    n_after_cells = int(after_mask.sum().item())
    if n_after_cells == 0:
        raise ValueError(
            "No targeted perturbed cells found for x_after training. "
            "Check obsm['gRNA_counts'] in the h5ad or lower --unperturbed-grna-threshold."
        )

    dataset_kwargs = {
        "x_expr": data["x_expr"][after_mask],
        "c_grna": data["c_grna"][after_mask],
        "dominant_grna_id": data["dominant_grna_id"][after_mask],
        "dominant_target_gene_id": data["dominant_target_gene_id"][after_mask],
        "x_baseline_pool": baseline_pool,
    }

    print(f"Training x_after cells: targeted_perturbed={n_after_cells}")

    dataset = PreparedPerturbSeqDataset(
        **dataset_kwargs,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cfg = ModelConfig(
        n_genes=n_genes,
        n_grnas=n_grnas,
        n_targets=int(data["n_targets"]),
        latent_dim=latent_dim,
        grna_embed_dim=grna_embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_weighted_grna_counts=use_weighted_grna_counts,
        grna_count_normalization=grna_count_normalization,
        top_guide_mask_k=top_guide_mask_k,
        guide_dropout_prob=guide_dropout_prob,
        grl_lambda=grl_lambda,
        alpha_guide_residual=alpha_guide_residual,
        cell_state_dim=cell_state_dim,
        gamma_cell_state=gamma_cell_state,
        grl_lambda_cell_state=grl_lambda_cell_state,
    )

    weights = LossWeights(
        rna=loss_rna,
        kl=loss_kl,
        adv_grna_classifier=loss_adv_grna_classifier,
        guide_residual_l2=loss_guide_residual_l2,
        cell_state_kl=loss_cell_state_kl,
        cell_state_target_adversary=loss_cell_state_target_adversary,
    )

    model = GRIT(cfg, grna_to_target_id=data["grna_to_target_id"])
    model = train_model(
        model=model,
        train_loader=loader,
        weights=weights,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
    )
    cell_level_output = output_path_from_prefix(Path(output_prefix), "cell_level_results.tsv")
    GRIT_score_metadata_output = output_path_from_prefix(
        Path(output_prefix), "GRIT_score_metadata.tsv"
    )
    expression_delta_output = output_path_from_prefix(
        Path(output_prefix), "x_pred_delta.tsv"
    )
    cell_level_df = write_cell_level_results(
        model=model,
        data=data,
        after_mask=after_mask,
        baseline_pool=baseline_pool,
        output_path=cell_level_output,
        metadata_output_path=GRIT_score_metadata_output,
        expression_delta_output_path=expression_delta_output,
        batch_size=batch_size,
    )
    print(f"Wrote cell-level results: {cell_level_output}")
    print(f"Wrote GRIT score metadata: {GRIT_score_metadata_output}")
    print(f"Wrote predicted expression delta: {expression_delta_output}")

    response_de_output = output_path_from_prefix(
        Path(output_prefix),
        "perturbation_response_de_genes.tsv",
    )
    write_perturbation_response_de_genes(
        cell_level_df=cell_level_df,
        expression_values=data["x_expr"][after_mask].cpu().numpy(),
        gene_names=data["gene_names"],
        output_path=response_de_output,
        spline_df=response_spline_df,
        include_guide_covariate=response_include_guide_covariate,
        min_cells_per_target=response_min_cells_per_target,
        min_unique_scores=response_min_unique_scores,
        min_shrunken_score_iqr=response_min_shrunken_score_iqr,
    )
    print(f"Wrote perturbation-response DE results: {response_de_output}")

    de_like_output = output_path_from_prefix(
        Path(output_prefix),
        "target_gene_de_like_genes.tsv",
    )
    de_like_metadata_output = output_path_from_prefix(
        Path(output_prefix),
        "target_gene_de_like_metadata.tsv",
    )
    write_target_gene_de_like_genes(
        data=data,
        after_mask=after_mask,
        reference_mask=reference_mask,
        nt_reference_mask=nt_reference_mask,
        unperturbed_reference_mask=unperturbed_reference_mask,
        reference_mode=reference_mode,
        output_path=de_like_output,
        metadata_output_path=de_like_metadata_output,
        seed=seed,
    )
    print(f"Wrote target-gene DE-like results: {de_like_output}")
    print(f"Wrote target-gene DE-like metadata: {de_like_metadata_output}")
    return model


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    """
    Example with the previous settings:

        python model_a_perturbation_transition_pytorch.py \
            --latent-dim 32 \
            --grna-embed-dim 32 \
            --hidden-dim 128 \
            --dropout 0.1 \
            --batch-size 64 \
            --n-epochs 20 \
            --learning-rate 0.001 \
            --output-prefix grit \
            --reference-mode nt_and_unperturbed \
            --unperturbed-grna-threshold 0.0 \
            --grna-count-normalization log1p \
            --top-guide-mask-k 3 \
            --guide-dropout-prob 0.05 \
            --multi-guide-qc true \
            --min-dominant-guide-fraction 0.5 \
            --max-second-guide-fraction 0.3 \
            --grl-lambda 0.2 \
            --alpha-guide-residual 0.1 \
            --cell-state-dim 8 \
            --gamma-cell-state 0.3 \
            --grl-lambda-cell-state 0.2 \
            --loss-rna 1.0 \
            --loss-kl 1e-3 \
            --loss-adv-grna-classifier 0.02 \
            --loss-guide-residual-l2 1e-3 \
            --loss-cell-state-kl 1e-3 \
            --loss-cell-state-target-adversary 0.02 \
            --response-spline-df 4 \
            --response-include-guide-covariate true \
            --response-min-shrunken-score-iqr 0.05
    """
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    config: Dict[str, object] = {}
    if args.config_file is not None and args.config_file.exists():
        with open(args.config_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is not None:
            config = dict(loaded)
        print(f"Loaded config: {args.config_file}")
    elif args.config_file is not None:
        print(
            f"Config file {args.config_file} not found; "
            "falling back to run_real_data_training defaults."
        )

    cli_keys = {"h5ad_file", "guide_map_file", "output_prefix"}
    overlapping = cli_keys.intersection(config.keys())
    if overlapping:
        raise ValueError(
            f"Config file must not contain CLI-only keys: {sorted(overlapping)}. "
            "Set these on the command line instead."
        )

    run_real_data_training(
        h5ad_file=args.h5ad_file,
        guide_map_file=args.guide_map_file,
        output_prefix=args.output_prefix,
        **config,
    )
