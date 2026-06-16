**Spec: perturbseq-agent**

Purpose
- Precise examples and CLI snippets for running `model/` functions and the GRIT model.

Entry points
- `prepare_perturb_h5ad.py`: dataset preparation helper.
- `model_a_perturbation_transition_pytorch.py`: primary model implementation (GRIT-compatible entrypoints).
- `run_model.sh`: example wrapper script to run training/evaluation.

Common commands
- Create venv and activate (see `setup.md`).
- Run the dataset prep (smoke test):

```bash
python perturbseq/model/prepare_perturb_h5ad.py --help
```

- Run the model wrapper:

```bash
bash perturbseq/model/run_model.sh
```

Validation
- After env setup, run `python -c "import torch; import scanpy as sc"` to validate core libs.

Read-only policy and wrapper entrypoints
- `perturbseq/model/` is read-only. Do not edit files in that directory.
- All pipeline drivers, plotting utilities, GSEA wrappers, and GEO downloaders live under `perturbseq/analysis/` or `perturbseq/agent/`.
- Example wrapper commands:

```bash
# download GEO datasets and optionally build h5ad from 10x matrices
python perturbseq/analysis/xx.script/01.download_geo.py --series GSE142078, GSE278572, GSE208240, GSE252965, GSE272457, GSE280506, GSE311503 --outdir perturbseq/00.data --build-h5ad

# run full pipeline (data prep -> GRIT -> downstream analysis)
python perturbseq/analysis/xx.script/03.run_pipeline.py --h5ad path/to/input.h5ad --output-prefix perturbseq/01.result

# generate plots from a completed run
python perturbseq/analysis/xx.script/##.plotting.py --results perturbseq/01.result/{GSE}_cell_level_results.tsv --outdir perturbseq/01.result/
```

These wrappers import functions from `perturbseq/model/` but never modify them.
