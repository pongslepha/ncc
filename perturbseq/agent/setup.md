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
