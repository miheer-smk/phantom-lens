<h1 align="center">Phantom Lens (PRISM)</h1>

<p align="center"><b>Physics-Anchored Deepfake Detection</b><br>
Miheer Satish Kulkarni — IIIT Nagpur · Supervisor: Dr. Nileshchandra K. Pikle</p>

---

## ⚠️ Correction notice — this `main` branch is superseded

**The results previously advertised on this branch are RETIRED.** An identity-overlap between the training and
test partitions (the same person's videos appeared in both) inflated the headline numbers. The following figures
are **no longer valid and must not be cited**:

- In-distribution FaceForensics++ AUCs of **0.9999 / 0.9991** (FaceSwap / NeuralTextures)
- Cross-dataset Celeb-DF-v2 AUC of **0.6867**
- The "**13/957 missed DeepFakes**" hard-negative figure
- The reproducer scripts on this branch (`results/exp1/run_exp1.py`, `results/exp_celebdf/run_celebdf_eval.py`)
  train and test with the leaked split and should **not** be run.

The work was re-done with strict **identity-disjoint** splits and a sealed, pre-registered evaluation.

## Where the corrected work lives

| Branch | Contents |
|---|---|
| [`major-revision`](../../tree/major-revision) | **Frozen resubmission state** — the corrected, leak-free results as submitted to the journal. |
| [`outstanding-results`](../../tree/outstanding-results) | **Latest** — the 196-D order-statistic representation and the sealed cross-dataset evaluation. |

**Corrected headline (identity-disjoint, sealed, pre-registered):** Celeb-DF-v2 sealed-test AUC **0.713
[0.687, 0.746]** (zero-shot). See `outstanding-results` → `196D_FINAL/` and `LOCKED_NUMBERS_196D.md` for the full
provenance, the freeze document, and the sealed-evaluation record.

> Nothing is merged into `main`; this branch is kept only as a pointer. Start from `major-revision` or
> `outstanding-results`.

---

*Corrected 2026-07-31. The retired numbers above are listed explicitly so prior readers can identify what changed.*
