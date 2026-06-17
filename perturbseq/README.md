# perturbseq

Analysis scaffolding around the GRIT perturbation-transition model. The
workflow takes heterogeneous GEO Perturb-seq datasets, reshapes them into a
canonical 10X form, builds a preprocessed `.h5ad`, runs the GRIT model, and
produces downstream plots and gene-set enrichment.

## Layout

```
perturbseq/
├── model/                 # read-only, authoritative model code
│   ├── prepare_perturb_h5ad.py            # 10X dir + guide_map.csv -> preprocessed .h5ad
│   ├── model_a_perturbation_transition_pytorch.py
│   ├── model_a_config.yaml
│   ├── run_model.sh                       # GRIT entrypoint
│   └── compute_guide_consistency_summary.py  # QC: do same-gene gRNAs give consistent signatures?
├── analysis/
│   ├── xx.script/         # analysis-side scripts (everything you run)
│   │   ├── 01.download_geo.py             # format-aware GEO downloader
│   │   ├── 02.prepare_h5ad.py             # reshape -> 10X + guide_map, run model prepare
│   │   ├── 03.inspect_data.py             # raw/processed data inspection and report generation
│   │   ├── 04.check_guide_matrix.py       # processed guide matrix sanity checks
│   │   ├── 10.run_pipeline.py             # optional end-to-end run wrapper
│   ├── 00.data/           # raw/ downloads + processed/ 10X inputs + logs/ (per GSE)
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

The numbered scripts under `analysis/xx.script/` run in order, but the workflow
is flexible: raw GEO downloads are stored under `analysis/00.data/`, processed
per-sample 10X inputs and `.h5ad` outputs are written under `analysis/01.result/`,
and logs are recorded under `analysis/00.data/logs/`.

### 1. Download — `01.download_geo.py`

The nine target series (GSE142078, GSE157977, GSE208240, GSE236057, GSE252965,
GSE272457, GSE278572, GSE280506, GSE311503) do **not** share a layout. The
downloader is format-aware: it inspects each series' GEO `suppl/` listing and
selects the files needed for each series' raw input shape.

One series is downloaded for reference but does not feed the canonical
`prepare_perturb_h5ad.py` pipeline:
- **GSE252965** — ATAC-seq only (no gene+guide matrix); has no preparation rule
  and is skipped.

**GSE157977** (mouse in-vivo perturb-seq) *is* handled, via the `dialout` shape,
but requires a manually built `guide_map.csv` (`grna,target_gene,
perturbation_barcode`) from the paper's Table S5: the deposited dial-out CSVs
carry perturbation barcodes but no protospacer→gene reference (GFP control → NT).
Without that `guide_map.csv` the series is skipped with an anomaly entry. See the
`dialout` shape below.

```bash
python perturbseq/analysis/xx.script/01.download_geo.py \
    --series GSE311503,GSE278572 \
    --outdir perturbseq/analysis/00.data

python perturbseq/analysis/xx.script/01.download_geo.py --series all \
    --outdir perturbseq/analysis/00.data --extract
```

Each series lands under `perturbseq/analysis/00.data/<GSE>/`.

### 2. Reshape + prepare — `02.prepare_h5ad.py`

This script reshapes heterogeneous raw inputs into a canonical per-sample 10X
directory that `perturbseq/model/prepare_perturb_h5ad.py` consumes. This is a
**format-only** step — matrices are reshaped and written as integer counts; no
statistical normalization (CP10K / log1p / z-score) and no QC filtering happen
here (those are done by the model script in step 2's `--run-prepare`). The
canonical output shape is:

```
<out-root>/<GSE>/<sample>/
    barcodes.tsv.gz
    features.tsv.gz     # id, name, feature_type (incl. CRISPR Guide Capture)
    matrix.mtx.gz       # features x cells, Gene Expression + CRISPR rows
    guide_map.csv       # grna,target_gene  (non-targeting controls -> NT)
```

The script auto-selects an input strategy (*shape*) per series. Each shape
builds the canonical 10X dir differently, and the guide-calling thresholds
passed to the model are tuned to that shape's guide-UMI characteristics:

- **`combined_triple`** — `GSE278572`, `GSE311503`, `GSE272457`. Already
  multi-feature 10X triples; re-emitted as clean combined triples. The guide
  matrix is the *raw* CRISPR-capture block (ambient-heavy), so these use
  `--guide-detection-umi 3` to recover confident single-guide cells.
- **`lookup`** — `GSE142078`, `GSE208240`, `GSE280506`. A GEX-only matrix plus a
  per-cell guide-assignment CSV (`Cell_Guide_Lookup.csv` / `cell_identities.csv`);
  a CRISPR Guide Capture block is synthesized from the author's per-cell calls.
  **Real per-guide UMIs are used when the CSV deposits a count column**
  (`UMI_count` / `read_count` / `Count`) and fall back to 1 UMI/guide otherwise.
  Because the calls are already author-curated, these use `--min-guide-umi 1`
  (model default `--guide-detection-umi 1`, single-guide filter on).
  *Note:* `GSE142078` is mixed — Run1 has no count column (1 UMI/guide
  synthesized) while Run2/Run3 ship a `Count` column (real UMIs); the source
  used is logged per run, and synthesized runs also raise an anomaly entry.
- **`metadata_guide_matrix`** — `GSE236057`. Non-standard `Counts.mtx` +
  `GeneNames.tsv` + `Barcodes.tsv` GEX plus a `Metadata.csv` that embeds a wide
  boolean guide-by-cell matrix; the synthesized CRISPR block is boolean
  (1 per assigned guide), so it also uses `--min-guide-umi 1`.
- **`dialout`** — `GSE157977`. GEX-only per-sample `.h5` plus a dial-out
  perturbation-barcode UMI CSV; the PBC→gene map comes from a manually built
  `guide_map.csv` (Table S5). Dial-out UMIs are low-depth, so these use
  `--guide-detection-umi 3 --min-guide-umi 3`.
- **`h5_multifeature`** / **`split_gex_guide`** — generic overrides for a single
  `.h5` that already contains CRISPR features, or separate GEX and guide
  matrices merged on shared cell barcodes.

Per-series rules also derive each gRNA's target gene and detect NT controls.
Because guide UMIs are real for some samples and synthesized (all-1) for others,
**`guide_umi` / `gRNA_counts` are not comparable across samples** and should not
be used as a quantitative feature across datasets.

When `--run-prepare` is provided, `02.prepare_h5ad.py` also invokes the
read-only `perturbseq/model/prepare_perturb_h5ad.py` on each reshaped sample
dir and writes the resulting `.h5ad` files into `perturbseq/analysis/01.result/`.
The model script is where the actual normalization happens (`sc.pp.normalize_total`
→ `log1p` → `regress_out` → `scale`), along with QC filtering and guide calling.
Incompatible series are logged and skipped with anomaly entries.

```bash
python perturbseq/analysis/xx.script/02.prepare_h5ad.py \
    --series GSE311503 \
    --data-root  perturbseq/analysis/00.data \
    --out-root   perturbseq/analysis/00.data \
    --run-prepare \
    --result-root perturbseq/analysis/01.result
```

### 3. Inspect processed data — `03.inspect_data.py`

Run this script after preparation to verify raw and processed datasets.
It generates `data_inspection.tsv` and `data_inspection_report.md` under
`perturbseq/analysis/00.data/logs/`, including:
- per-sample cell and feature counts
- split counts for Gene Expression vs CRISPR Guide Capture
- guide count, distinct target gene count, and NT guide presence
- guide capture and MOI statistics
- dominance warnings for overly pervasive guides

```bash
python perturbseq/analysis/xx.script/03.inspect_data.py \
    --root perturbseq/analysis/00.data --deep
```

### 4. Validate guide matrices — `04.check_guide_matrix.py`

This script checks processed 10X directories under `perturbseq/analysis/00.data/processed/`
or similar reshaped paths and validates that the CRISPR Guide Capture block is
present and consistent with `guide_map.csv`.
It writes a human-readable log to
`perturbseq/analysis/00.data/logs/guide_matrix_check.log`.

```bash
python perturbseq/analysis/xx.script/04.check_guide_matrix.py \
    --root perturbseq/analysis/00.data
```

### 5. Optional wrapper — `10.run_pipeline.py`

The optional wrapper runs a supplied `.h5ad` through GRIT and downstream
plotting. It prefers `perturbseq/model/run_model.sh` and falls back to
`perturbseq/model/model_a_perturbation_transition_pytorch.py` if needed.

```bash
python perturbseq/analysis/xx.script/10.run_pipeline.py \
    --h5ad perturbseq/analysis/01.result/<sample>.h5ad \
    --output-prefix perturbseq/output/myrun \
    --run-grit
```

## Environment

See `agent/setup.md` for creating the `perturb` virtualenv and
`agent/env_packages.md` for the recorded package set. Core dependencies:
`scanpy`, `anndata`, `scipy`, `pandas`, `numpy`, `seaborn`, `matplotlib`,
`python-igraph`, `leidenalg`, and `gseapy` (for GSEA). The agent's role and
operating contract are described in `agent/agent.md`.
