# ⛔ DEPRECATED — leakage-inflated results. DO NOT USE.

Everything in this directory produced **retired, data-leakage-inflated numbers** and is kept only
for provenance/audit. **None of these results, scripts, or numbers are valid.** They must not be
cited, run, or reported.

## Why these are retired
The original pipeline trained and evaluated on the **same** manipulation CSVs (test fakes were present
in the training set) and did **not** use identity-disjoint splitting. This inflated the reported AUCs.
The corrected, leakage-free pipeline (identity-disjoint official FF++ 720/140/140, seed 42) lives in
the repository root and supersedes everything here.

## Retired numbers that appear in this folder (all INVALID)
| Retired | Where | Honest replacement |
|---|---|---|
| FF++ 0.9939 / 0.9991 / 0.9999 | `results/exp1`, `results/exp3`, README (old) | per-manip 53-D **0.978 / 0.969 / 0.875 / 0.905**, mean **0.932** |
| Celeb-DF 0.6989 / 0.6867 | `results/exp_celebdf`, `gate_*` | **0.632** [0.613, 0.654] |
| Hard-negative 13/957 (1.36%) | `results/exp5` | clean **17/133 (12.78%)** identity-disjoint |

## Contents
- `results/` — the entire original results tree (exp1, exp2, exp3, exp5, exp_celebdf, per-manip and
  cross-manip dirs, visualizations, old reports). Leaky.
- `gate_scripts/` — `gate_exp1.py`, `gate_celebdf.py`: recon-phase scripts whose **targets were the
  retired published numbers** (e.g. 0.9999 / 0.9991 / 0.6867). Not part of the honest pipeline.
- `gate_outputs/` — their run logs.

## Where the honest work is
- **`LOCKED_NUMBERS.md`** (`Major Revision Results/07_summary/`) — single source of truth, all numbers with provenance.
- **`AUTHOR_HANDOFF.md`** — facts/tables/decisions for the authors.
- **`results_clean/`** — all current, leakage-free result tables.
- **`celebdf_reconciliation.md`** — why 0.6989 ≠ 0.6867 ≠ 0.632.
- **`REPRODUCE.md`** (repo root) — clean regeneration commands.
