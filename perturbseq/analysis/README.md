Analysis scaffolding for perturbseq

Policy
- `perturbseq/model/` is read-only. Do not modify files under that directory.
- Add helper scripts under `perturbseq/analysis/` and `perturbseq/agent/`.

Files
- `download_geo.py`: download GEO series and optional AnnData building from 10x matrices.
- `run_pipeline.py`: driver wrapper that shells out to model scripts and runs downstream plotting.
- `plotting.py`: produces bar, volcano, and heatmap plots from GRIT outputs.
- `gsea.py`: simple wrapper to run GSEA via `gseapy`.

Usage examples
```bash
# download GEO datasets and optionally build h5ad from 10x matrices
python perturbseq/analysis/download_geo.py --series GSE142078,GSE278572 --outdir perturbseq/data --build-h5ad

# run full pipeline (if model entrypoints are present)
python perturbseq/analysis/run_pipeline.py --output-prefix perturbseq/output/myrun --run-grit

# plotting only
python perturbseq/analysis/plotting.py --results perturbseq/output/myrun_cell_level_results.tsv --outdir perturbseq/output/plots

# GSEA
python perturbseq/analysis/gsea.py --gene-list my_ranks.tsv --outdir perturbseq/output/gsea
```
