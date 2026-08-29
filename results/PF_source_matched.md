# Item F — DF40 with source-matched reals

**Outcome: the correction is real but immaterial. Macro moves 0.7717 → 0.7741, +0.0024.**

Artifacts: `results/PF_source_matched.json`

---

## F1 — What actually formed each method's real class

| Method | fakes from `ff` | fakes from `cdf` | dominant source |
|---|---|---|---|
| InSwapper | 112 | 406 | cdf |
| FaceDancer | 134 | 614 | cdf |
| Wav2Lip | 134 | 644 | cdf |
| SadTalker | 132 | 653 | cdf |
| HyperReenact | 140 | 641 | cdf |

**The pooled real class behind the 0.7717 macro was 134 FF++ + 798 Celeb-DF = 932, i.e. 86%
Celeb-DF.** Every method is cdf-dominated (79–83% of its fakes), so the pooled real class was in
fact *roughly* aligned with the dominant fake source — which is why the correction turns out small.
But the `ff`-derived fakes in each method were being ranked against a real class that is 86% from a
different corpus, which is the defect the item identified.

## A methodological correction I had to make mid-item

My first source-matched implementation paired each subset with its own reals and then **concatenated
the scores into one AUC**. That reproduced the pooled number exactly (0.7717) — and the reason is
instructive: **AUC is a global ranking statistic, so a single pooled AUC still compares every `ff`
fake against every Celeb-DF real regardless of how the pairs were assembled.** Concatenation does not
achieve source matching.

The correct computation is to evaluate **AUC separately within each source** and then combine.
Reported below fake-count weighted; the unweighted variant is in the JSON.

## F2 / F3 — Three configurations side by side

| Method | pooled reals | Celeb-DF reals throughout | **source-matched** |
|---|---|---|---|
| InSwapper | 0.7519 | 0.7427 | **0.7566** |
| FaceDancer | 0.6738 | 0.6624 | **0.6766** |
| Wav2Lip | 0.5969 | 0.5844 | **0.5995** |
| SadTalker | 0.9709 | 0.9696 | **0.9719** |
| HyperReenact | 0.8649 | 0.8594 | **0.8659** |
| **macro** | **0.7717** (sd 0.149) | **0.7637** (sd 0.154) | **0.7741** (sd 0.148) |

**What each measures:**

- **pooled reals (0.7717)** — every fake ranked against a real class that is 86% Celeb-DF.
  Mixes source and manipulation; the `ff` fakes are scored largely against out-of-source reals.
- **Celeb-DF reals throughout (0.7637)** — source-matched for the `cdf` subsets, deliberately
  mismatched for the `ff` subsets. Useful as the *lower* bound on the acquisition contribution.
- **source-matched (0.7741)** — each subset ranked only against reals from its own corpus, AUCs
  computed independently and combined. **Isolates manipulation.** This is the primary reported number.

## F4 — Is the move material?

**No. +0.0024 macro AUC**, well inside any reasonable materiality threshold and two orders of
magnitude below the 0.149 spread across methods. Every per-method change is between +0.0010 and
+0.0047, and **no ordering changes**.

Per the brief's instruction that this result "gets more scrutiny, not less", the manuscript states
the correction explicitly rather than silently adopting the better number:

> **Source matching.** DF40 fakes derive from two source corpora. In our initial evaluation each
> method's fakes were ranked against a pooled real class (86% Celeb-DF), so subsets derived from
> FaceForensics++ were partly ranked against out-of-source reals. We recomputed with each subset
> ranked only against reals from its own source corpus, computing the AUC within each source and
> combining. The macro-average moves from 0.772 to 0.774 (+0.002) and no per-method ordering changes;
> we report the source-matched figure as primary and the pooled and Celeb-DF-only variants as
> sensitivity analyses.

**Why the correction is small, stated so the reader can check it:** every DF40 method is
cdf-dominated (79–83% of fakes from `cdf`), so the 86%-Celeb-DF pooled real class was already close
to source-matched for the bulk of each method's fakes. The `ff` subsets, which were genuinely
mismatched, are the minority.

## Interaction with item G

The source-matched configuration is **more robust to the dataset-identity confound** than the pooled
one, because within each source both classes come from the same corpus, so the corpus-identity
signal measured in item G (AUC 0.675 among fakes) cannot align with the label. This is a second
reason to prefer it as primary, independent of its numerical value.
