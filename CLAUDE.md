# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Churn prediction pipeline for the public IBM Telco Customer Churn dataset (7043 rows). Currently only the data pipeline stages exist (ETL + load to warehouse); no model training yet.

## Environment

- Python virtualenv at `.venv/` (Windows). Activate with `source .venv/Scripts/activate` (bash) or `.venv\Scripts\Activate.ps1` (PowerShell).
- Two requirements files:
  - `requirements.txt` — runtime deps for the pipeline scripts in `src/` (pandas, duckdb). Keep this scoped to what `src/*.py` actually imports.
  - `requirements-dev.txt` — adds notebook/EDA tooling (matplotlib, seaborn, ipykernel, nbformat, nbconvert, nbclient) on top of `requirements.txt` via `-r requirements.txt`.
- Install: `.venv/Scripts/python.exe -m pip install -r requirements-dev.txt` (use `requirements.txt` alone for a prod-like/runtime-only install).

## Commands

Run scripts with the venv interpreter explicitly (no activation needed):

```
.venv/Scripts/python.exe src/etl.py
.venv/Scripts/python.exe src/load.py
```

Execute the EDA notebook end-to-end and check for errors:

```
.venv/Scripts/python.exe -m nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
```

Run the test suite:

```
.venv/Scripts/python.exe -m pytest
```

There is no linter or build step configured yet.

## Pipeline architecture

Two sequential, idempotent stages, each a standalone script with a `main()` guarded by `if __name__ == "__main__":`. Run order matters: `etl.py` must run before `load.py`.

1. **`src/etl.py`** — `data/raw/Telco-Customer-Churn.csv` → `data/processed/telco_clean.csv`
   - Coerces `TotalCharges` (string with 11 blanks) to numeric; asserts the blanks correspond exactly to `tenure == 0` before filling them with 0 (not mean imputation — this is a deliberate domain decision, not a missing-data shortcut).
   - Maps `Churn` Yes/No → 1/0.
   - Creates `data/processed/` via `Path(...).mkdir(parents=True, exist_ok=True)` before writing.

2. **`src/load.py`** — `data/processed/telco_clean.csv` → `data/warehouse/telco.duckdb` (table `customers`)
   - Uses `CREATE OR REPLACE TABLE` so re-running never duplicates rows.
   - Asserts the loaded table has exactly 7043 rows and that the churn rate computed in DuckDB matches the churn rate computed from the source CSV in pandas — a cross-check that the load didn't silently corrupt data.
   - Creates `data/warehouse/` before connecting.

When extending either script, preserve this pattern: validate with asserts against known-good values immediately after the transformation/load, and ensure the destination directory exists before writing (`Path(path).parent.mkdir(parents=True, exist_ok=True)`), rather than assuming `data/` subfolders exist.

## Data directory

- `data/raw/` — source CSV, gitignored.
- `data/processed/` — ETL output, gitignored.
- `data/warehouse/` — DuckDB warehouse, gitignored.
- `notebooks/eda.ipynb` — exploratory analysis over `data/processed/telco_clean.csv`; not wired into any pipeline script, run manually.

Only code and dependency manifests are committed; all of `data/` is excluded via `.gitignore`.

## Reglas de trabajo

- Antes de escribir código, explicar el plan en 3-4 líneas.
- Commits chicos con Conventional Commits, uno por cambio.
- No hacer `git push` — lo hace el usuario desde su propia terminal.
- No cambiar nada más de lo que se pide.

## Git conventions

Commits follow Conventional Commits (`feat:`, `fix:`, `chore:`). Remote is GitLab (`origin`), default branch `main`. Push requires a GitLab personal access token (HTTPS basic auth with a password no longer works) — if a push hangs, it's waiting on an interactive credential prompt that won't surface in a non-interactive shell; the user needs to run it themselves or configure a credential helper.
