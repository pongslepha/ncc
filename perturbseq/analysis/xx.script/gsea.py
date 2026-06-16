#!/usr/bin/env python3
"""GSEA wrapper: run ORA/GSEA using gseapy if available, otherwise provide instructions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import subprocess

def run_gsea_with_gseapy(gene_list: Path, outdir: Path) -> int:
    # gene_list: simple two-column TSV (gene, score) expected by many tools
    try:
        import gseapy as gp
    except Exception:
        print("gseapy not installed. Install with: pip install gseapy")
        return 1
    # Minimal example: prerank
    ranks = gene_list
    outdir.mkdir(parents=True, exist_ok=True)
    print("Running prerank GSEA with gseapy (gene_list must be two-column gene,score)")
    gp.prerank(rnk=str(ranks), outdir=str(outdir), format="png")
    return 0

def main() -> None:
    parser = argparse.ArgumentParser(description="GSEA wrapper")
    parser.add_argument("--gene-list", type=Path, required=True, help="Two-column gene,score file (TSV)")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for GSEA results")
    args = parser.parse_args()
    rc = run_gsea_with_gseapy(args.gene_list, args.outdir)
    if rc != 0:
        sys.exit(rc)

if __name__ == "__main__":
    main()
