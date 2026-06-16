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

The GEO series do not all ship that shape. This script reshapes each
into a normalized per-sample directory (raw data is read from
<data-root>/raw/<GSE>/, written under <out-root>/processed/<GSE>/<sample>/):

    <out-root>/processed/<GSE>/<sample>/
        barcodes.tsv.gz
        features.tsv.gz      # 3 cols: id, name, feature_type (incl. CRISPR)
        matrix.mtx.gz        # genes x cells, Gene Expression + CRISPR rows
        guide_map.csv        # columns: grna,target_gene  (NT controls -> "NT")

Input shapes handled (auto-selected by --series, override with --shape):
  combined_triple  GSE278572, GSE311503, GSE272457
                   already multi-feature; we re-emit a clean triple + guide_map.
  lookup           GSE142078, GSE208240, GSE280506
                   GEX-only matrix (triple or .h5) + a per-cell guide CSV
                   (Cell_Guide_Lookup.csv or cell_identities.csv); a CRISPR
                   Guide Capture block is SYNTHESIZED from the per-cell calls.
  metadata_guide_matrix  GSE236057
                   GEX-only matrix under non-standard names (Counts.mtx +
                   GeneNames.tsv + Barcodes.tsv) plus a Metadata.csv that embeds
                   a WIDE boolean guide-by-cell matrix (Enh*/Pos_*/Neg_* cols);
                   a CRISPR Guide Capture block is SYNTHESIZED from the TRUE
                   cells (Neg_* -> NT, Pos_<GENE> -> <GENE>, Enh<N>_* -> Enh<N>).
  h5_multifeature  (generic override) read_10x_h5(gex_only=False) when a single
                   .h5 already contains CRISPR Guide Capture features.
  split_gex_guide  (generic override) separate GEX and gRNA matrices merged on
                   shared barcodes.

Incompatible series (downloaded by 01 for reference, but skipped here):
  GSE157977        guides recorded only as protospacer sequences in a per-sample
                   dial-out UMI CSV; no protospacer->gene reference and no NT
                   label are deposited, so guide_map / the required NT control
                   cannot be derived. See INCOMPATIBLE_SERIES below.

Usage:
    python 02.prepare_h5ad.py --series GSE311503 \
        --data-root  perturbseq/analysis/00.data \
        --out-root   perturbseq/analysis/00.data \
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

import geo_utils
from geo_utils import find_triples, load_triple_adata

NT = "NT"  # canonical non-targeting label expected by prepare_perturb_h5ad.py
GUIDE_FEATURE_TYPE = "CRISPR Guide Capture"
GENE_FEATURE_TYPE = "Gene Expression"

# Per-series context for logging / anomaly trail, set in main() each iteration.
_CTX: dict = {"root": None, "series": None, "log": None}


def _anomaly(title: str, observation: str, action: str) -> None:
    if _CTX.get("root") is None:
        return
    geo_utils.append_anomaly(
        _CTX["root"], _CTX["series"], title, observation, action, logger=_CTX["log"]
    )

MODEL_PREPARE = (
    Path(__file__).resolve().parents[2] / "model" / "prepare_perturb_h5ad.py"
)

# ---------------------------------------------------------------------------
# Per-series rules: input shape, guide-name -> gene parsing, NT detection,
# and any extra args to hand to prepare_perturb_h5ad.py.
# ---------------------------------------------------------------------------
SERIES_RULES: dict[str, dict] = {
    # Real Perturb-seq UMI matrices carry heavy ambient-guide contamination:
    # with guide_detection_umi=1 the median cell "detects" ~13 guides, so the
    # single-guide (max_guides_detected<=1) filter drops nearly everything.
    # Raising the detection threshold to 3 UMIs recovers the real single-guide
    # cells (validated on GSE311503 D1: 5 -> ~1.3k cells). This is an
    # analysis-side knob exposed by the model script; the model is unchanged.
    "GSE278572": {
        "shape": "combined_triple",
        "nt_regex": r"(?i)non.?target",
        "strip_regex": r"_\d+_CRISPRi$",
        "mt_pattern": r"^MT-",
        "prepare_args": ["--guide-detection-umi", "3"],
    },
    "GSE311503": {
        "shape": "combined_triple",
        "nt_regex": r"(?i)no.?target|non.?target",
        "strip_regex": r"[._]\d+$",
        "mt_pattern": r"^MT-",
        "prepare_args": ["--guide-detection-umi", "3"],
    },
    "GSE272457": {
        "shape": "combined_triple",
        "nt_regex": r"(?i)^nt[._-]",
        "strip_regex": r"_\d+$",
        "mt_pattern": r"^MT-",  # human+mouse mix; symbols stay human-cased
        "prepare_args": ["--guide-detection-umi", "3"],
    },
    # GSE280506 & GSE208240: the deposited count matrix (series triple AND the
    # .h5 for 280506; the nested filtered matrix for 208240) is Gene Expression
    # ONLY -- guides live in cell_identities.csv (per-cell guide_identity). So
    # both are the "lookup" shape, not h5_multifeature / split_gex_guide.
    "GSE280506": {
        "shape": "lookup",
        "nt_regex": r"(?i)non.?target|negative.?control|safe.?harbor|scramble",
        "gene_parse": "strand",   # e.g. ZNF335_ZNF335_+_44600782.23-P1P2 -> ZNF335
        "mt_pattern": r"^MT-",
        "prepare_args": ["--min-guide-umi", "1"],
    },
    "GSE208240": {
        "shape": "lookup",
        "nt_regex": r"(?i)non.?target|negative.?control|safe.?harbor|scramble",
        "gene_parse": "strand",   # e.g. SMOC1_+_70346211.23-P1P2_CR1-cs1 -> SMOC1
        "mt_pattern": r"^MT-",
        "prepare_args": ["--min-guide-umi", "1"],
    },
    "GSE142078": {
        "shape": "lookup",
        "nt_regex": r"(?i)^non.?targeting",
        "strip_regex": r"_G?\d+$",   # e.g. CHD8_G3 -> CHD8
        "mt_pattern": r"^MT-",
        # Synthetic guide counts are 1 UMI/cell, so relax the UMI threshold.
        "prepare_args": ["--min-guide-umi", "1"],
    },
    # GSE236057: GEX matrix under non-standard names + a Metadata.csv embedding a
    # WIDE boolean guide-by-cell matrix. Guide columns are Enh<N>_g<M>_chr...
    # (enhancer targets), Pos_<GENE> (positive controls) and Neg_* (negative /
    # non-targeting controls). The synthesized CRISPR block is boolean (1 per
    # assigned guide), so relax the UMI threshold like the other lookup series.
    "GSE236057": {
        "shape": "metadata_guide_matrix",
        "nt_regex": r"(?i)^neg",
        "gene_parse": "enh_pos_neg",
        "mt_pattern": r"^MT-",
        "prepare_args": ["--min-guide-umi", "1"],
    },
}

# Series that 01 downloads for reference but that CANNOT feed the canonical
# Perturb-seq pipeline; main() skips them with an explanatory anomaly entry.
INCOMPATIBLE_SERIES: dict[str, str] = {
    "GSE157977": (
        "Guides are recorded only as protospacer SEQUENCES in a per-sample "
        "dial-out UMI CSV; GEO deposits no protospacer->gene reference and no "
        "non-targeting (NT) label, so guide_map (grna->target_gene) and the NT "
        "control that prepare_perturb_h5ad.py requires cannot be derived."
    ),
}


# ---------------------------------------------------------------------------
# Guide map derivation
# ---------------------------------------------------------------------------
def _gene_from_strand(name: str) -> str:
    """Target gene from a strand-encoded guide name, e.g.
    'SMOC1_+_70346211.23-P1P2' -> 'SMOC1', 'ZNF335_ZNF335_+_..' -> 'ZNF335'."""
    head = re.split(r"_[+-]_", name)[0]
    toks = head.split("_")
    if len(toks) >= 2 and toks[0] == toks[1]:
        return toks[0]
    return head


def _gene_from_enh_pos_neg(name: str) -> str:
    """Target from a GSE236057-style guide name. Positive controls are
    'Pos_<GENE>' (-> '<GENE>'); enhancer guides are 'Enh<N>_g<M>_chr..' or the
    bare 'Enh<N>' (-> 'Enh<N>', the enhancer locus). (Neg_* is handled by the
    nt_regex before this is reached.)"""
    if name.startswith("Pos_"):
        return name[len("Pos_"):]
    return re.split(r"_g\d+", name)[0]


def derive_gene(guide: str, rule: dict) -> str:
    """Map a gRNA name to its target gene; NT controls collapse to "NT"."""
    if re.search(rule["nt_regex"], guide):
        return NT
    parse = rule.get("gene_parse", "strip")
    if parse == "strand":
        gene = _gene_from_strand(guide)
    elif parse == "enh_pos_neg":
        gene = _gene_from_enh_pos_neg(guide)
    else:
        gene = re.sub(rule.get("strip_regex", r"$"), "", guide)
    return gene or guide


def build_guide_map(guide_names: list[str], rule: dict) -> pd.DataFrame:
    rows = [
        {"grna": g, "target_gene": derive_gene(g, rule)}
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
        _anomaly(
            "No non-targeting (NT) guides detected",
            observation=(
                f"nt_regex {rule['nt_regex']!r} matched 0 of {len(df)} guides; "
                f"examples: {guide_names[:6]}. prepare_perturb_h5ad.py needs >=1 NT."
            ),
            action=(
                "Inspect the guide naming scheme and update this series' nt_regex "
                "in SERIES_RULES so non-targeting controls map to 'NT'."
            ),
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
# find_triples / load_triple_adata are shared via geo_utils.
# ---------------------------------------------------------------------------
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
# Shape: lookup  (GEX-only matrix + per-cell guide assignment CSV)
#   GSE142078 -> *_Cell_Guide_Lookup.csv (CellBarcode,sgRNA; bare barcodes)
#   GSE208240 -> cell_identities.csv (guide_identity; paired/multiplet, "-N" gemgroup)
#   GSE280506 -> cell_identities.csv (guide_identity; "*" = unassigned, "-N" gemgroup)
# A CRISPR Guide Capture block is synthesized (1 UMI per assigned guide); the
# model's single-guide / min-UMI filters then drop multiplet cells.
# ---------------------------------------------------------------------------
_UNASSIGNED = {"", "*", "n/a", "na", "none", "nan", "unassigned"}


def _load_h5(h5: Path):
    a = sc.read_10x_h5(h5, gex_only=False)
    a.var_names_make_unique()
    return a


def _find_lookup_csvs(series_dir: Path) -> list[Path]:
    for pat in ("*Cell_Guide_Lookup.csv*", "*cell_identities.csv*",
                "*[Gg]uide*ookup*.csv*", "*identities*.csv*"):
        hits = sorted(series_dir.rglob(pat))
        if hits:
            return hits
    return []


def _read_lookup(lookup_path: Path) -> pd.DataFrame:
    lk = pd.read_csv(lookup_path)
    bc_col = next((c for c in lk.columns if re.search(r"(?i)barcode|cell", c)), lk.columns[0])
    # prefer an explicit guide-call column, then a generic guide/sgRNA column
    sg_col = next((c for c in lk.columns if re.search(r"(?i)guide_identity|feature_call", c)), None)
    if sg_col is None:
        sg_col = next((c for c in lk.columns if re.search(r"(?i)sgrna|guide|grna|protospacer", c)),
                      lk.columns[-1])
    out = lk[[bc_col, sg_col]].copy()
    out.columns = ["barcode", "guide"]
    out["barcode"] = out["barcode"].astype(str)
    out["guide"] = out["guide"].astype(str)
    return out


def _match_lookup(sample: str, lookups: list[Path]) -> Path:
    """Pick the lookup CSV for a sample: shared token (e.g. 'Run1') else the
    single/first available."""
    if len(lookups) == 1:
        return lookups[0]
    for token in reversed(sample.split("_")):  # try most-specific token first
        hit = next((l for l in lookups if token and token in l.name), None)
        if hit:
            return hit
    return lookups[0]


def prep_lookup(series_dir: Path, out_root: Path, rule: dict) -> list[Path]:
    lookups = _find_lookup_csvs(series_dir)
    if not lookups:
        print(f"  [WARN] no guide lookup / cell_identities CSV under {series_dir}")
        _anomaly("No per-cell guide assignment file found",
                 observation=f"Searched {series_dir} for Cell_Guide_Lookup/cell_identities CSV.",
                 action="This 'lookup' series needs a per-cell guide CSV; check the download.")
        return []

    # GEX sources: each 10X triple is a sample; if none, fall back to a single .h5.
    triples = find_triples(series_dir)
    samples: list[tuple[str, object]] = []
    if triples:
        for sample, parts in triples.items():
            samples.append((sample, load_triple_adata(parts)))
    else:
        h5s = sorted(series_dir.rglob("*.h5"))
        if not h5s:
            print(f"  [WARN] no GEX matrix (triple or .h5) under {series_dir}")
            return []
        sample = re.sub(r"_filtered_feature_bc_matrix$", "", h5s[0].stem)
        samples.append((sample, _load_h5(h5s[0])))

    out_dirs = []
    for sample, gex in samples:
        lookup = _match_lookup(sample, lookups)
        print(f"  sample '{sample}'  GEX={gex.shape}  + lookup '{lookup.name}'")
        try:
            out_dirs.append(_build_from_lookup(gex, sample, lookup, out_root / sample, rule))
        except Exception as exc:  # noqa: BLE001 - log and continue with other samples
            print(f"  [WARN] lookup build failed for {sample}: {exc}")
            _anomaly("Lookup build failed",
                     observation=f"sample {sample}: {exc}",
                     action="Check barcode overlap / guide column parsing for this sample.")
    return out_dirs


def _build_from_lookup(
    gex, sample: str, lookup_path: Path, out_dir: Path, rule: dict
):
    lk = _read_lookup(lookup_path)

    # split multi-guide calls (";" separated) into one row per (barcode, guide)
    lk = lk.assign(guide=lk["guide"].str.split(";")).explode("guide")
    lk["guide"] = lk["guide"].str.strip()
    lk = lk[~lk["guide"].str.lower().isin(_UNASSIGNED)]

    # match barcodes to the GEX matrix: try as-is, then fall back to stripping a
    # trailing "-N" gem-group/lane suffix only if that improves overlap (it would
    # otherwise collide cells from different gem groups).
    gex_bc = pd.Index(gex.obs_names.astype(str))
    overlap_raw = lk["barcode"].isin(gex_bc).mean() if len(lk) else 0.0
    lk_stripped = lk.assign(barcode=lk["barcode"].str.replace(r"-\d+$", "", regex=True))
    gex_bc_stripped = pd.Index(gex.obs_names.astype(str).str.replace(r"-\d+$", "", regex=True))
    overlap_strip = lk_stripped["barcode"].isin(gex_bc_stripped).mean() if len(lk) else 0.0
    if overlap_strip > overlap_raw and gex_bc_stripped.is_unique:
        gex.obs_names = gex_bc_stripped
        gex_bc, lk = gex_bc_stripped, lk_stripped
        print(f"  (stripped '-N' barcode suffix to match: {overlap_strip:.0%} overlap)")

    lk = lk[lk["barcode"].isin(gex_bc)]
    print(f"  lookup covers {lk['barcode'].nunique():,} / {gex.n_obs:,} GEX cells, "
          f"{lk['guide'].nunique():,} distinct guides")
    if lk.empty:
        raise ValueError("No lookup barcodes intersect the GEX matrix barcodes.")

    guide_names = sorted(lk["guide"].unique())
    guide_index = {g: i for i, g in enumerate(guide_names)}
    bc_pos = {bc: i for i, bc in enumerate(gex.obs_names.astype(str))}

    # synthesize a guides x cells count matrix: 1 UMI per assigned guide
    rows = [guide_index[g] for g in lk["guide"]]
    cols = [bc_pos[bc] for bc in lk["barcode"]]
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
# Shape: metadata_guide_matrix  (GSE236057)
#   GEX-only matrix under non-standard names:
#     *_Counts.mtx.gz       (genes x cells, integer UMI counts)
#     *_GeneNames.tsv.gz    (3 cols: name, ensembl_id, "Gene Expression")
#     *_Barcodes.tsv.gz     (one cell barcode per line)
#   plus *_Metadata.csv.gz whose columns 0=barcode, then per-cell QC, then a
#   WIDE boolean guide-by-cell block (Enh*/Pos_*/Neg_* columns, "TRUE"/"FALSE").
#   A CRISPR Guide Capture block is synthesized (1 per TRUE guide call).
# ---------------------------------------------------------------------------
def _read_lines_gz(path: Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def _find_one(series_dir: Path, key: str) -> Path | None:
    """First file under series_dir whose lowercased name contains ``key``."""
    hits = sorted(p for p in series_dir.rglob("*") if p.is_file() and key in p.name.lower())
    return hits[0] if hits else None


def prep_metadata_guide_matrix(series_dir: Path, out_root: Path, rule: dict) -> list[Path]:
    counts = _find_one(series_dir, "counts.mtx")
    genes_f = _find_one(series_dir, "genenames")
    bc_f = _find_one(series_dir, "barcodes")
    meta_f = _find_one(series_dir, "metadata")
    missing = [n for n, p in [("Counts.mtx", counts), ("GeneNames", genes_f),
                              ("Barcodes", bc_f), ("Metadata", meta_f)] if p is None]
    if missing:
        print(f"  [WARN] missing files for metadata_guide_matrix: {missing}")
        _anomaly("Missing named-matrix files",
                 observation=f"under {series_dir}: missing {missing}",
                 action="Re-run 01.download_geo.py for this series.")
        return []

    # --- GEX matrix, oriented genes x cells ---
    with gzip.open(counts, "rb") as fh:
        mat = sparse.csr_matrix(scipy_io.mmread(fh))
    barcodes = _read_lines_gz(bc_f)
    gene_rows = [ln.split("\t") for ln in _read_lines_gz(genes_f)]
    n_genes, n_cells = len(gene_rows), len(barcodes)
    if mat.shape == (n_genes, n_cells):
        gex = mat
    elif mat.shape == (n_cells, n_genes):
        gex = mat.T.tocsr()
    else:
        raise ValueError(
            f"Counts shape {mat.shape} matches neither genes x cells "
            f"({n_genes} x {n_cells}) nor its transpose."
        )
    gex = gex.astype(np.int64)
    # GeneNames columns are (name, ensembl_id, feature_type) -- note the
    # name/id order is reversed vs a standard 10X features.tsv (id, name, type).
    gene_name = [r[0] for r in gene_rows]
    gene_id = [r[1] if len(r) > 1 else r[0] for r in gene_rows]
    print(f"  GEX {gex.shape[0]:,} genes x {gex.shape[1]:,} cells")

    # --- wide boolean guide block from the metadata CSV ---
    header = list(pd.read_csv(meta_f, nrows=0).columns)
    bc_col = header[0]
    guide_cols = [c for c in header if re.match(r"(?i)^(enh|pos|neg)", str(c))]
    if not guide_cols:
        raise ValueError("No Enh*/Pos_*/Neg_* guide columns found in metadata.")
    meta = pd.read_csv(
        meta_f, usecols=[bc_col] + guide_cols,
        dtype={c: "category" for c in guide_cols},
    )
    meta_bc = meta[bc_col].astype(str).tolist()
    bc_pos = {bc: i for i, bc in enumerate(barcodes)}

    g_rows: list[int] = []
    g_cols: list[int] = []
    n_unmatched = 0
    for j, c in enumerate(guide_cols):
        cats = list(meta[c].cat.categories)
        if "TRUE" not in cats:
            continue
        codes = meta[c].cat.codes.to_numpy()
        for r in np.nonzero(codes == cats.index("TRUE"))[0]:
            pos = bc_pos.get(meta_bc[r])
            if pos is None:
                n_unmatched += 1
            else:
                g_rows.append(j)
                g_cols.append(pos)
    if n_unmatched:
        print(f"  [note] {n_unmatched:,} guide calls referenced barcodes absent "
              "from the GEX matrix (ignored).")
    guide_mat = sparse.csr_matrix(
        (np.ones(len(g_rows), dtype=np.int64), (g_rows, g_cols)),
        shape=(len(guide_cols), n_cells),
    )
    n_assigned = int((np.asarray(guide_mat.sum(axis=0)).ravel() > 0).sum())
    print(f"  guide block: {len(guide_cols):,} guides x {n_cells:,} cells; "
          f"{n_assigned:,} cells carry >=1 guide call")

    features = pd.DataFrame({
        "id": gene_id + guide_cols,
        "name": gene_name + guide_cols,
        "feature_type": [GENE_FEATURE_TYPE] * n_genes
        + [GUIDE_FEATURE_TYPE] * len(guide_cols),
    })
    matrix = sparse.vstack([gex, guide_mat]).tocsr()
    guide_map = build_guide_map(guide_cols, rule)
    sample = re.split(r"_[Cc]ounts", counts.name)[0]
    out_dir = out_root / sample
    write_10x_dir(out_dir, matrix, features, barcodes, guide_map)
    return [out_dir]


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
    "lookup": prep_lookup,
    "metadata_guide_matrix": prep_metadata_guide_matrix,
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
    log = geo_utils.get_logger(args.data_root, "prepare")
    log.info("Preparing %d series; raw <- %s, processed -> %s",
             len(series_list), geo_utils.raw_root(args.data_root),
             geo_utils.processed_root(args.out_root))

    for series in series_list:
        if series in INCOMPATIBLE_SERIES:
            reason = INCOMPATIBLE_SERIES[series]
            log.warning("\n=== %s === incompatible; skipping prepare. %s", series, reason)
            _CTX.update(root=args.data_root, series=series, log=log)
            _anomaly(
                "Series not Perturb-seq-pipeline-compatible — skipped",
                observation=reason,
                action="01 keeps the raw download for reference; prepare is skipped. "
                       "Supply the missing reference (e.g. a protospacer->gene map "
                       "with NT controls) to enable this series.",
            )
            summary.append((series, "incompatible (skipped)"))
            continue
        if series not in SERIES_RULES:
            log.warning("\n=== %s === no rule configured; skipping.", series)
            summary.append((series, "no rule"))
            continue
        rule = SERIES_RULES[series]
        shape = args.shape or rule["shape"]
        series_dir = geo_utils.raw_series_dir(args.data_root, series)
        _CTX.update(root=args.data_root, series=series, log=log)
        log.info("\n=== %s === shape='%s'  raw=%s", series, shape, series_dir)
        if not series_dir.exists():
            log.warning("  [WARN] %s not found; run 01.download_geo.py first.", series_dir)
            summary.append((series, "no data"))
            continue

        # Normalized 10X dirs go under <out-root>/processed/<GSE>/<sample>/.
        out_root = geo_utils.processed_root(args.out_root) / series
        normalized = SHAPE_DISPATCH[shape](series_dir, out_root, rule)
        if not normalized:
            summary.append((series, "no output"))
            continue

        made_h5ad = 0
        if args.run_prepare:
            result_root = args.result_root or (args.out_root / "01.result" / series)
            for input_dir in normalized:
                rc = run_prepare(input_dir, result_root, rule)
                h5ad_path = result_root / f"{input_dir.name}.h5ad"
                if rc == 0 and h5ad_path.exists():
                    made_h5ad += 1
                    _check_survival(h5ad_path)
                else:
                    log.warning("  [WARN] prepare failed (rc=%s) for %s", rc, input_dir)
                    _anomaly(
                        "prepare_perturb_h5ad.py failed",
                        observation=f"return code {rc}; no h5ad at {h5ad_path}.",
                        action="Check the prepare log above (e.g. QC thresholds, "
                               "guide_map mismatch) and re-run on this sample.",
                    )
        summary.append(
            (series, f"{len(normalized)} 10X dir(s)" + (f", {made_h5ad} h5ad" if args.run_prepare else ""))
        )

    log.info("\n=== summary ===")
    for series, status in summary:
        log.info("  %s: %s", series, status)


def _check_survival(h5ad_path: Path, *, min_cells: int = 50) -> None:
    """Flag a degenerate prepare result (very few cells, or all one target)."""
    try:
        adata = sc.read_h5ad(h5ad_path)
    except Exception as exc:  # noqa: BLE001
        _anomaly("Could not re-open prepared h5ad",
                 observation=f"{h5ad_path}: {exc}", action="Inspect the file manually.")
        return
    n = adata.n_obs
    if "gene" in adata.obs.columns and n:
        vc = adata.obs["gene"].value_counts()
        top_gene, top_n = (vc.index[0], int(vc.iloc[0])) if len(vc) else ("?", 0)
        n_targets = int(adata.obs["gene"].nunique())
    else:
        top_gene, top_n, n_targets = "?", 0, 0
    if n < min_cells:
        _anomaly(
            f"Very low cell survival after prepare ({n} cells)",
            observation=(
                f"{h5ad_path.name} kept only {n} cells across {n_targets} target(s). "
                "Usually means the single-guide / UMI QC filters are too strict for "
                "this dataset's guide-UMI distribution."
            ),
            action="Raise --guide-detection-umi and/or lower --min-guide-umi/"
                   "--min-nfeature-rna for this series in SERIES_RULES.prepare_args.",
        )
    elif n_targets <= 1 and n:
        _anomaly(
            f"All surviving cells map to a single target ({top_gene})",
            observation=(
                f"{h5ad_path.name}: {top_n}/{n} cells are '{top_gene}'; no target "
                "diversity. Often caused by one pathologically dominant guide present "
                "in nearly every cell (see 03.inspect_data.py guide-dominance check)."
            ),
            action="Inspect the guide-UMI distribution; consider excluding the "
                   "over-represented guide before guide calling.",
        )


if __name__ == "__main__":
    main()
