**Spec: perturbseq-agent**

Purpose
- Precise examples and CLI snippets for running `model/` functions and the GRIT model.

Entry points
- `perturbseq/analysis/xx.script/01.download_geo.py`: raw GEO downloader into `perturbseq/analysis/00.data`.
- `perturbseq/analysis/xx.script/02.prepare_h5ad.py`: data normalization and `h5ad` preparation for `perturbseq/analysis/01.result`.
- `perturbseq/analysis/xx.script/03.inspect_data.py`: raw/processed dataset inspection and report writer.
- `perturbseq/analysis/xx.script/04.check_guide_matrix.py`: guide matrix validation for processed 10X directories.
- `prepare_perturb_h5ad.py`: dataset preparation helper in the read-only model directory.
- `model_a_perturbation_transition_pytorch.py`: primary model implementation (GRIT-compatible entrypoints).
- `run_model.sh`: example wrapper script to run training/evaluation.

Common commands
- Create venv and activate (see `setup.md`).
- Run the dataset prep (smoke test):

```bash
python perturbseq/model/prepare_perturb_h5ad.py --help
```

- Run the canonical data workflow:

```bash
python perturbseq/analysis/xx.script/01.download_geo.py --series GSE142078,GSE208240,GSE236057,GSE272457,GSE278572,GSE280506,GSE311503 --outdir perturbseq/analysis/00.data --extract
python perturbseq/analysis/xx.script/02.prepare_h5ad.py --series GSE142078,GSE208240,GSE236057,GSE272457,GSE278572,GSE280506,GSE311503 --data-root perturbseq/analysis/00.data --out-root perturbseq/analysis/00.data --run-prepare --result-root perturbseq/analysis/01.result
python perturbseq/analysis/xx.script/03.inspect_data.py --root perturbseq/analysis/00.data --deep
python perturbseq/analysis/xx.script/04.check_guide_matrix.py --root perturbseq/analysis/00.data
```

- Run the model wrapper:

```bash
bash perturbseq/model/run_model.sh
```

Validation
- After env setup, run `python -c "import torch; import scanpy as sc"` to validate core libs.
- When a check needs more than a one-liner, create a dedicated inspection script under `perturbseq/analysis/` or `perturbseq/agent/` and run it. For example, after downloading data write an `inspect_data.py` to load the matrices/`AnnData` and report shapes, cell/gene counts, `obsm['gRNA_counts']` presence, and NT vs. perturbed group sizes:

```bash
# inspect a downloaded / prepared dataset before running the pipeline
python perturbseq/analysis/xx.script/inspect_data.py --h5ad perturbseq/00.data/{GSE}.h5ad
```

- Such check scripts are inspection-only: they may import from `perturbseq/model/` but never modify it, and should print a clear pass/fail summary for easy verification.

Read-only policy and wrapper entrypoints
- `perturbseq/model/` is read-only. Do not edit files in that directory.
- All pipeline drivers, plotting utilities, GSEA wrappers, and GEO downloaders live under `perturbseq/analysis/` or `perturbseq/agent/`.
- Example wrapper commands:

```bash
# download GEO datasets (format-aware; --extract unpacks any *.tar archives)
python perturbseq/analysis/xx.script/01.download_geo.py \
    --series GSE142078,GSE157977,GSE208240,GSE236057,GSE252965,GSE272457,GSE278572,GSE280506,GSE311503 \
    --outdir perturbseq/analysis/00.data --extract
# (or --series all for every configured series)

# run full pipeline (data prep -> GRIT -> downstream analysis)
python perturbseq/analysis/xx.script/03.run_pipeline.py --h5ad path/to/input.h5ad --output-prefix perturbseq/01.result

# generate plots from a completed run
python perturbseq/analysis/xx.script/##.plotting.py --results perturbseq/01.result/{GSE}_cell_level_results.tsv --outdir perturbseq/01.result/
```

These wrappers import functions from `perturbseq/model/` but never modify them.
