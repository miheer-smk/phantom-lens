# Branch guide — phantom-lens

Four branches. Nothing is merged between them; each is a distinct, intentional state.

| branch | state | authoritative for | start here if… |
|---|---|---|---|
| **`major-revision`** | **Frozen** | ✅ **the journal resubmission** | you want the exact corrected, leak-free results as submitted |
| **`outstanding-results`** | Latest | the newest (196-D) cross-dataset work | you want the strongest cross-dataset result and its full record |
| `best-revision` | Working | development history of Track D/E | you want the exploratory trail behind `outstanding-results` |
| `main` | Retired / pointer | nothing — corrected elsewhere | (don't) — it only carries a correction notice |

## Details

### `major-revision` — authoritative for the resubmission (FROZEN)
The corrected, identity-disjoint, leakage-free results **as submitted to the journal**. This is the reference
state for the paper: 53-D PRISM, in-distribution FF++ mean AUC **0.932**, zero-shot Celeb-DF-v2 (official-style
full set) AUC **0.632**, plus the DeLong / Xception-baseline / missingness / hard-negative analyses. Do not
rebase or rewrite — it must match what the co-authors submitted.

### `outstanding-results` — latest cross-dataset work (196-D)
Post-submission improvement programme. Adds the **196-D order-statistic representation** and a **sealed,
pre-registered** cross-dataset evaluation: **Celeb-DF-v2 sealed-test AUC 0.713 [0.687, 0.746]** (see the branch
README and `196D_FINAL/`). **Important:** the 196-D sealed number uses a **custom identity-disjoint dev/test
split**, not Celeb-DF-v2's official protocol, and the sealed half is easier than the full set — so it is **not**
directly comparable to the `major-revision` 0.632 or to published Celeb-DF figures. The valid, like-for-like
claims are the sealed-test deltas (+0.030 over 53-D, +0.056 over 50-D). Everything here is dev-selected then
evaluated once on the sealed half; 57 dev evaluations, sealed budget 1 (spent).

### `best-revision` — working branch
The development branch from which `outstanding-results` was cut (same tip commit at creation). Holds the full
Track D/E exploration. Content mirrors `outstanding-results` minus the branch-specific README/BRANCHES updates.

### `main` — retired
The original pre-leak-fix branch. Its former headline numbers (0.9999 / 0.9991 / 0.6867 / 13-957) are **retired**
due to a train/test identity overlap. Its README now carries only a correction notice pointing here. No code from
any branch is merged into `main`.

## Which number goes in the paper?
- **In-distribution and the primary Celeb-DF number:** `major-revision` (frozen resubmission).
- **The 196-D representation gain and sealed cross-dataset result:** `outstanding-results`, quoted with its caveats.
- The two are **different configurations and different Celeb-DF protocols** — present them as such, never as a single
  before/after on the same axis.
