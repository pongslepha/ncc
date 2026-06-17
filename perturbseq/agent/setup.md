**Setup: create and verify `perturb` virtualenv**

1) Create the virtualenv:

```bash
python3 -m venv ~/venvs/perturb
source ~/venvs/perturb/bin/activate
```

2) Upgrade pip and install basics:

```bash
pip install --upgrade pip setuptools wheel
```

3) Install packages discovered by the agent (placeholder list in `env_packages.md`). Example packages often required:

```bash
pip install numpy scipy pandas scikit-learn scanpy anndata torch torchvision matplotlib seaborn h5py GEOparse gseapy
```

4) Verify imports (smoke tests):

```bash
python - <<'PY'
import sys
modules = ["numpy","scipy","pandas","scanpy","anndata","torch"]
failed = []
for m in modules:
    try:
        __import__(m)
    except Exception as e:
        failed.append((m,str(e)))
print('FAILED' if failed else 'OK', failed)
PY
```

5) Record installed packages:

```bash
pip freeze > perturbseq/agent/env_packages.md
```

6) Run quick model smoke test:

```bash
python perturbseq/model/prepare_perturb_h5ad.py --help
bash perturbseq/model/run_model.sh
```

7) Run the canonical analysis workflow using the repository wrappers:

```bash
python perturbseq/analysis/xx.script/01.download_geo.py \
    --series GSE142078,GSE208240,GSE236057,GSE272457,GSE278572,GSE280506,GSE311503 \
    --outdir perturbseq/analysis/00.data --extract

python perturbseq/analysis/xx.script/02.prepare_h5ad.py \
    --series GSE142078,GSE208240,GSE236057,GSE272457,GSE278572,GSE280506,GSE311503 \
    --data-root perturbseq/analysis/00.data \
    --out-root perturbseq/analysis/00.data \
    --run-prepare \
    --result-root perturbseq/analysis/01.result

python perturbseq/analysis/xx.script/03.inspect_data.py --root perturbseq/analysis/00.data --deep
python perturbseq/analysis/xx.script/04.check_guide_matrix.py --root perturbseq/analysis/00.data
```

8) Review the logs and outputs:

- `perturbseq/analysis/00.data/logs/` for raw/processed inspection reports and anomalies.
- `perturbseq/analysis/01.result/` for generated `.h5ad` files, QC plots, and result tables.
