#!/usr/bin/env python3
"""Driver wrapper: data prep -> optional GRIT run -> downstream analysis.

This script is a non-destructive wrapper that imports/executes model scripts
or shells out to them; it never modifies files under `perturbseq/model/`.

Usage examples:
  python perturbseq/analysis/run_pipeline.py --h5ad data/input.h5ad --output-prefix perturbseq/output/myrun --run-grit

The script expects the model outputs to be written under <output-prefix> with
the canonical filenames described in perturbseq/agent/agent.md.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
import os

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def run_data_prep(h5ad_path: Path) -> None:
    # If the user supplied an h5ad, we assume data prep is not required.
    print(f"Using provided h5ad: {h5ad_path}")

def run_grit_wrapper(output_prefix: Path) -> int:
    script = Path("perturbseq/model/run_model.sh")
    if script.exists():
        cmd = ["bash", str(script), str(output_prefix)]
        print("Running GRIT via:", " ".join(cmd))
        return subprocess.call(cmd)
    # fallback: try calling the python module if present
    py = Path("perturbseq/model/model_a_perturbation_transition_pytorch.py")
    if py.exists():
        cmd = [sys.executable, str(py), "--output-prefix", str(output_prefix)]
        print("Running GRIT via:", " ".join(cmd))
        return subprocess.call(cmd)
    print("No GRIT entrypoint found under perturbseq/model/; skipping GRIT run.")
    return 0

def run_downstream(output_prefix: Path, h5ad: Path | None) -> None:
    # Load results if available and call plotting/gsea wrappers
    plots_dir = output_prefix.parent / "plots"
    ensure_dir(plots_dir)
    # call plotting module
    cmd = [sys.executable, "perturbseq/analysis/plotting.py", "--results", f"{output_prefix}_cell_level_results.tsv", "--outdir", str(plots_dir)]
    print("Calling plotting:", " ".join(cmd))
    subprocess.call(cmd)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run perturbseq pipeline wrapper")
    parser.add_argument("--h5ad", type=Path, default=None, help="Optional preprocessed h5ad input")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Output prefix for results (directory + prefix) e.g. perturbseq/output/myrun")
    parser.add_argument("--run-grit", action="store_true", help="Run GRIT model using the model/ entrypoint")
    args = parser.parse_args()

    out_prefix: Path = args.output_prefix
    ensure_dir(out_prefix.parent)

    if args.h5ad is not None:
        run_data_prep(args.h5ad)

    if args.run_grit:
        rc = run_grit_wrapper(out_prefix)
        if rc != 0:
            print("GRIT run exited with code", rc)
            sys.exit(rc)

    run_downstream(out_prefix, args.h5ad)

if __name__ == "__main__":
    main()
