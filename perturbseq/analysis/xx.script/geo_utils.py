#!/usr/bin/env python3
"""Shared helpers for the GEO Perturb-seq analysis scripts (01/02/03).

Centralizes:
  * directory layout       <root>/raw/<GSE>/ , <root>/processed/<GSE>/<sample>/
  * logging                <root>/logs/pipeline.log (+ console)
  * the anomaly reasoning log
      <root>/logs/anomalies.md  (append_anomaly: what was weird + how we
      handled it -- the "reasoning trail" requested by the user)
  * 10X triple discovery / loading reused by 02 and 04.

Lives under perturbseq/analysis/; never touches perturbseq/model/.
"""
from __future__ import annotations

import gzip
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import scanpy as sc

RAW = "raw"
PROCESSED = "processed"
LOGS = "logs"


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def raw_root(root: Path) -> Path:
    return Path(root) / RAW


def processed_root(root: Path) -> Path:
    return Path(root) / PROCESSED


def logs_root(root: Path) -> Path:
    path = Path(root) / LOGS
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_series_dir(root: Path, series: str) -> Path:
    """Resolve a series' raw dir, tolerating both <root>/raw/<GSE> and the
    legacy flat <root>/<GSE> layout."""
    new = raw_root(root) / series
    if new.exists():
        return new
    legacy = Path(root) / series
    return legacy if legacy.exists() else new


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(root: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logs_root(root) / "pipeline.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Anomaly reasoning log
# ---------------------------------------------------------------------------
_ANOMALY_HEADER = """# Data anomaly & reasoning log

This file is the running trail of "something looked wrong -> here is what and
why -> here is how the code handled it". Design-time findings are seeded at the
top; the analysis scripts (01/02/04) append timestamped runtime entries below.

"""


def append_anomaly(
    root: Path,
    series: str,
    title: str,
    observation: str,
    action: str,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Append a structured anomaly entry to <root>/logs/anomalies.md."""
    path = logs_root(root) / "anomalies.md"
    if not path.exists():
        path.write_text(_ANOMALY_HEADER, encoding="utf-8")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n## [{ts}] {series} — {title}\n"
        f"- **Observed:** {observation}\n"
        f"- **Action:** {action}\n"
    )
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(block)
    if logger is not None:
        logger.warning("ANOMALY [%s] %s | observed: %s | action: %s",
                        series, title, observation, action)


# ---------------------------------------------------------------------------
# 10X triple discovery / loading (shared by 02 and 04)
# ---------------------------------------------------------------------------
_TRIPLE_SUFFIX_KEY = {
    "_barcodes.tsv.gz": "barcodes",
    "_features.tsv.gz": "features",
    "_genes.tsv.gz": "features",  # legacy CellRanger v2
    "_matrix.mtx.gz": "matrix",
    "barcodes.tsv.gz": "barcodes",
    "features.tsv.gz": "features",
    "genes.tsv.gz": "features",
    "matrix.mtx.gz": "matrix",
}


def find_triples(series_dir: Path) -> dict[str, dict[str, Path]]:
    """Group *_barcodes/_features|genes/_matrix files by their sample prefix."""
    samples: dict[str, dict[str, Path]] = {}
    for path in sorted(Path(series_dir).rglob("*")):
        if not path.is_file():
            continue
        for suffix, key in _TRIPLE_SUFFIX_KEY.items():
            if path.name.endswith(suffix):
                prefix = path.name[: -len(suffix)].rstrip("_") or path.parent.name
                samples.setdefault(prefix, {})[key] = path
                break
    return {
        sample: parts
        for sample, parts in samples.items()
        if {"barcodes", "matrix"} <= parts.keys()
    }


def _n_columns_gz(path: Path) -> int:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.strip():
                return len(line.rstrip("\n").split("\t"))
    return 0


def _stage_features_v3(src: Path, dst: Path) -> None:
    """Stage a feature file as a 3-column v3 features.tsv.gz. Legacy CellRanger
    v2 genes.tsv has only (id, name); append a 'Gene Expression' feature_type
    column so scanpy reads it in v3 mode (avoids brittle legacy auto-detection)."""
    if _n_columns_gz(src) >= 3:
        dst.write_bytes(src.read_bytes())
        return
    with gzip.open(src, "rt") as fin, gzip.open(dst, "wt") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) == 1:
                cols = [cols[0], cols[0]]
            fout.write(f"{cols[0]}\t{cols[1]}\tGene Expression\n")


def load_triple_adata(parts: dict[str, Path]) -> sc.AnnData:
    """Load a triple by staging it under standard 10X v3 names in a temp dir."""
    staging = Path(tempfile.mkdtemp(prefix="triple_"))
    (staging / "barcodes.tsv.gz").write_bytes(parts["barcodes"].read_bytes())
    (staging / "matrix.mtx.gz").write_bytes(parts["matrix"].read_bytes())
    if "features" in parts:
        _stage_features_v3(parts["features"], staging / "features.tsv.gz")
    return sc.read_10x_mtx(
        staging, var_names="gene_symbols", make_unique=True, gex_only=False
    )


def mtx_header_dims(matrix_path: Path) -> tuple[int, int, int] | None:
    """Cheaply read a MatrixMarket header -> (n_rows/features, n_cols/cells, nnz)
    without loading the matrix. Returns None if the header can't be parsed."""
    opener = gzip.open if str(matrix_path).endswith(".gz") else open
    try:
        with opener(matrix_path, "rt") as fh:
            for line in fh:
                if line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    return int(parts[0]), int(parts[1]), int(parts[2])
                return None
    except Exception:
        return None
    return None
