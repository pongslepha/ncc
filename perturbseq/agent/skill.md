**Skill: perturbseq-agent-skill**

Capabilities
- Inspect Python files under `model/` and extract imports and callable entrypoints.
- Prepare reproducible Python virtualenv `perturb` and install required packages.
- Execute smoke-tests and simple runs of GRIT-compatible model code.
- Build GEO data download wrappers for specified perturb-seq datasets and normalize them to `AnnData` if possible.
- Orchestrate the repository's analysis wrappers: `01.download_geo.py`, `02.prepare_h5ad.py`, `03.inspect_data.py`, and `04.check_guide_matrix.py`.

Constraints
- The agent treats `perturbseq/model/` as read-only: it will never modify files inside `model/`.
- All new code (wrappers, plotting, GSEA, GEO downloaders, helper functions) will be created under `perturbseq/analysis/` or `perturbseq/agent/` and import from `perturbseq/model/` as needed.

Inputs
- Path to repository root (defaults to project root).
- Optional: list of explicit packages to install.

Outputs
- `env_packages.md`: a recorded `pip freeze` after installation.
- Short run logs and results of smoke-tests.

Failure handling
- If an import fails, produce reproducible minimal reproduction code and suggested package to install.
