# perturbseq

Analysis scaffolding around the GRIT perturbation-transition model. The
workflow takes heterogeneous GEO Perturb-seq datasets, normalizes them into a
canonical 10X form, builds a preprocessed `.h5ad`, runs the GRIT model, and
produces downstream plots and gene-set enrichment.

## Layout

```
perturbseq/
├── model/                 # read-only, authoritative model code
│   ├── prepare_perturb_h5ad.py            # 10X dir + guide_map.csv -> preprocessed .h5ad
│   ├── model_a_perturbation_transition_pytorch.py
│   ├── model_a_config.yaml
│   └── run_model.sh                       # GRIT entrypoint
├── analysis/
│   ├── xx.script/         # analysis-side scripts (everything you run)
│   │   ├── 01.download_geo.py             # format-aware GEO downloader
│   │   ├── 02.prepare_h5ad.py             # normalize -> 10X + guide_map, run model prepare
│   │   ├── 03.run_pipeline.py             # GRIT run + downstream driver
│   │   ├── plotting.py                    # bar / volcano / heatmap from GRIT outputs
│   │   └── gsea.py                        # prerank GSEA via gseapy
│   ├── 00.data/           # downloaded + normalized data (per GSE)
│   └── 01.result/         # model prepare / GRIT outputs
└── agent/                 # agent role, setup, and environment docs
```

## Policy

- **`perturbseq/model/` is read-only.** Do not modify anything under it. The
  model script `prepare_perturb_h5ad.py` is strict: it reads a *single* 10X
  directory (combined Gene Expression + CRISPR Guide Capture matrix) plus a
  `guide_map.csv`. Reshaping inputs into that form is the job of the analysis
  scripts, never an edit to the model.
- All helpers, wrappers, and utilities live under `perturbseq/analysis/` and
  `perturbseq/agent/`.

## Pipeline

The numbered scripts under `analysis/xx.script/` run in order.

### 1. Download — `01.download_geo.py`

The nine target series (GSE142078, GSE157977, GSE208240, GSE236057, GSE252965,
GSE272457, GSE278572, GSE280506, GSE311503) do **not** share a layout. The
downloader is format-aware: it inspects each series' GEO `suppl/` listing and
picks the right files per series "kind" (legacy CellRanger triples, combined
multi-feature triples, `filtered_feature_bc_matrix.h5`, separate GEX+guide
matrices, a non-standard-named GEX matrix + guide-in-metadata, or a big
`*.tar`/`*.tar.gz`).

Two series are downloaded but **cannot** feed the canonical pipeline:
- **GSE252965** — ATAC-seq only (no gene+guide matrix); skipped at download.
- **GSE157977** — mouse perturb-seq whose guides are recorded only as
  protospacer *sequences* in per-sample dial-out UMI CSVs, with no
  protospacer→gene reference and no non-targeting (NT) label deposited. The
  `RAW.tar` is downloaded for reference, but `02`/`03` skip it (the missing
  guide→gene map / NT control makes `guide_map.csv` underivable). Both
  exclusions are recorded in `00.data/logs/anomalies.md`.

```bash
python analysis/xx.script/01.download_geo.py \
    --series GSE311503,GSE278572 \
    --outdir analysis/00.data

# every configured series; extract any downloaded tar archives
python analysis/xx.script/01.download_geo.py --series all \
    --outdir analysis/00.data --extract
```

Each series lands under `<outdir>/<GSE>/`.

### 2. Normalize + prepare — `02.prepare_h5ad.py`

Reshapes each downloaded series into the canonical per-sample 10X directory the
model expects:

```
<out-root>/<GSE>/<sample>/
    barcodes.tsv.gz
    features.tsv.gz     # id, name, feature_type (incl. CRISPR Guide Capture)
    matrix.mtx.gz       # features x cells, Gene Expression + CRISPR rows
    guide_map.csv       # grna,target_gene  (non-targeting controls -> "NT")
```

Input shapes are auto-selected per series (override with `--shape`):
`combined_triple`, `lookup`, `metadata_guide_matrix`, `h5_multifeature`,
`split_gex_guide`. `metadata_guide_matrix` (GSE236057) reads a non-standard
GEX matrix (`Counts.mtx` + `GeneNames.tsv` + `Barcodes.tsv`) and synthesizes a
CRISPR Guide Capture block from the wide boolean guide-by-cell matrix embedded
in `Metadata.csv` (`Neg_*`→NT, `Pos_<GENE>`→gene, `Enh<N>_*`→enhancer locus).
Per-series rules derive each gRNA's target gene, detect non-targeting (NT)
guides, and pass model knobs (e.g. `--guide-detection-umi 3` to handle ambient
guide contamination). With `--run-prepare` it then invokes the read-only
`model/prepare_perturb_h5ad.py` on each normalized dir. Pipeline-incompatible
series (GSE157977) are skipped here with an anomaly entry.

```bash
python analysis/xx.script/02.prepare_h5ad.py \
    --series GSE311503 \
    --data-root  analysis/00.data \
    --out-root   analysis/00.data/prepared \
    --run-prepare \
    --result-root analysis/01.result
```

### 3. Run GRIT + downstream — `03.run_pipeline.py`

Non-destructive wrapper that (optionally) runs the GRIT model via
`model/run_model.sh` and then calls the downstream plotting step. It never
modifies `model/`.

```bash
python analysis/xx.script/03.run_pipeline.py \
    --h5ad analysis/01.result/<sample>.h5ad \
    --output-prefix perturbseq/output/myrun \
    --run-grit
```

### Plotting & GSEA

`plotting.py` reads a GRIT results TSV (columns such as `target`,
`grit_score`, `gene`, `logfc`, `pval`) and writes bar, volcano, and heatmap
plots. `gsea.py` runs prerank GSEA via `gseapy` on a two-column `gene,score`
ranks file.

```bash
python analysis/xx.script/plotting.py \
    --results perturbseq/output/myrun_cell_level_results.tsv \
    --outdir  perturbseq/output/plots

python analysis/xx.script/gsea.py \
    --gene-list my_ranks.tsv \
    --outdir    perturbseq/output/gsea
```

## Environment

See `agent/setup.md` for creating the `perturb` virtualenv and
`agent/env_packages.md` for the recorded package set. Core dependencies:
`scanpy`, `anndata`, `scipy`, `pandas`, `numpy`, `seaborn`, `matplotlib`,
`python-igraph`, `leidenalg`, and `gseapy` (for GSEA). The agent's role and
operating contract are described in `agent/agent.md`.
