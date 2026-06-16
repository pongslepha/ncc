#!/usr/bin/env python3
"""Plotting utilities for GRIT outputs: bar, volcano, heatmap.

This module reads the per-cell results TSV produced by GRIT (or a compatible
table with columns including 'guide', 'target', 'grit_score', 'pval', 'logfc')
and writes publication-ready plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys


def read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Results file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, sep=None, engine="python")


def plot_grit_bar(df: pd.DataFrame, out: Path) -> None:
    # Expect per-target GRIT metadata with columns target, grit_score
    if df.empty:
        print("Empty dataframe; skipping bar plot")
        return
    if "target" in df.columns and "grit_score" in df.columns:
        agg = df.groupby("target")["grit_score"].median().sort_values(ascending=False)
        plt.figure(figsize=(8, max(4, len(agg) * 0.25)))
        sns.barplot(x=agg.values, y=agg.index, palette="vlag")
        plt.xlabel("GRIT score (median)")
        plt.tight_layout()
        plt.savefig(out / "grit_bar.png", dpi=200)
        plt.close()
    else:
        print("No 'target'/'grit_score' columns found; skipping bar plot")


def plot_volcano(df: pd.DataFrame, out: Path) -> None:
    # Expect columns 'logfc' and 'pval' or 'p_value'
    if df.empty:
        print("Empty dataframe; skipping volcano plot")
        return
    pcol = next((c for c in ("pval", "p_value", "p" ) if c in df.columns), None)
    lcol = next((c for c in ("logfc", "log2FoldChange", "lfc") if c in df.columns), None)
    namecol = next((c for c in ("gene","feature","name") if c in df.columns), None)
    if pcol is None or lcol is None:
        print("No p-value or log-fold-change columns found; skipping volcano")
        return
    df = df.copy()
    df["-log10p"] = -np.log10(df[pcol].clip(lower=1e-300))
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=lcol, y="-log10p", data=df, s=10, alpha=0.6)
    plt.axhline(-np.log10(0.05), color="grey", linestyle="--")
    plt.xlabel(lcol)
    plt.ylabel("-log10(p)")
    plt.tight_layout()
    plt.savefig(out / "volcano.png", dpi=200)
    plt.close()


def plot_heatmap(df: pd.DataFrame, out: Path, top_n: int = 50) -> None:
    # Heatmap of top responsive genes across targets. This expects a wide table
    # or a long table that can be pivoted to genes x targets with an effect size.
    if df.empty:
        print("Empty dataframe; skipping heatmap")
        return
    if "gene" in df.columns and "target" in df.columns and "logfc" in df.columns:
        pivot = df.pivot_table(index="gene", columns="target", values="logfc", aggfunc="median").fillna(0)
        top_genes = pivot.abs().max(axis=1).sort_values(ascending=False).head(top_n).index
        mat = pivot.loc[top_genes]
        plt.figure(figsize=(12, max(4, len(top_genes)*0.15)))
        sns.heatmap(mat, center=0, cmap="vlag", linewidths=0.5)
        plt.tight_layout()
        plt.savefig(out / "heatmap.png", dpi=200)
        plt.close()
    else:
        print("Required columns for heatmap not found (gene,target,logfc); skipping")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GRIT results: bar, volcano, heatmap")
    parser.add_argument("--results", type=Path, required=True, help="Path to per-gene or per-cell results TSV/CSV")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for plots")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = read_results(args.results)
    plot_grit_bar(df, args.outdir)
    plot_volcano(df, args.outdir)
    plot_heatmap(df, args.outdir)


if __name__ == "__main__":
    main()
