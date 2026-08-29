# P1.2 — Celeb-DF v2 count reconciliation

**Verdict.** The brief's hypothesis is **rejected**, and the truth is worse than it proposed. 806 was
not back-computed as *processed + excluded*. **806 is the authentic-video count retained by a
retired, superseded extraction pipeline.** The manuscript pairs an output count from the old pipeline
(806) with an output count from the current pipeline (798) and presents the difference as attrition.

The true input population is **890 authentic**, and the true authentic attrition is **92, not 8** —
understated by a factor of 11.5.

**The AUC 0.6322 is unaffected.** It is computed over the 6121 rows that do exist and reproduces
exactly (§P0.4). This is a reporting defect, not a results defect. See DEFECT-004.

---

## 1. Test of the brief's hypothesis: is 806 = 798 + 8?

Arithmetically yes. **Evidentially no.** The origin of 806 is on disk:

`legacy/phantomlens/archive/deprecated_leaky/results/exp_celebdf/results.json`
```
"n_test_real": 806,   "n_test_fake": 5323,
"cross_dataset_auc": 0.6866817952338486
```

806 is `n_test_real` — an **evaluated** count, not an input count — from the retired leakage-prone
pipeline (the 0.6867 run). The same 806 + 5323 pairing also underlies the other retired figure,
0.6989, per `Major Revision Results/07_summary/open_ambiguities.md:26`.

The prior authors already knew: `open_ambiguities.md:85` reads
> *"the ~1% face-detection survival difference (my CelebDF real=798 vs original 806)"*

806 and 798 are **two runs of the same stage**, differing in MediaPipe survival. Neither is an input
population. The "8 excluded" reading is an artifact of subtracting two incommensurable numbers.

Corroborating detail: the **fake** count is *identical* (5323) in both pipelines while only the real
count moves. A genuine input→output attrition of 8 would not leave 5,323 fakes unchanged to the unit
across a pipeline change that altered real survival.

## 2. Was the official test list used? Was YouTube-real included?

**Official test list: NO.** `data/celebdf_v2/List_of_testing_videos.txt` is present (518 entries:
340 fake + 178 real = 108 Celeb-real + 70 YouTube-real). The evaluation used **6121** videos —
11.8× the official test split. The published zero-shot result is over the **entire** Celeb-DF v2
release, not its designated test partition.

**YouTube-real: YES, included.** Determined by classifying all 6121 retained `video_path` values in
`features/celebdf_features.csv` against the three release folders:

| Folder | retained | label |
|---|---|---|
| Celeb-real | 545 | real |
| YouTube-real | **253** | real |
| Celeb-synthesis | 5323 | fake |
| **total** | **6121** | 798 real / 5323 fake |

All 6121 resolve to files on disk; zero unmatched.

## 3. The arithmetic, gap by gap

### Gap 1 — authentic: 890 on disk → 798 processed

```
  Celeb-real on disk                    590
  YouTube-real on disk                  300
                                     ------
  authentic input population            890      <-- the correct denominator

  Celeb-real retained                   545   (45 excluded,  7.6%)
  YouTube-real retained                 253   (47 excluded, 15.7%)
                                     ------
  authentic processed                   798
  authentic EXCLUDED                     92   (10.3%)
```
The manuscript's "806 authentic → 798 processed" implies **8** exclusions. The measured figure is
**92**. Note the attrition is itself uneven across sub-corpora: YouTube-real loses more than twice
the proportion Celeb-real does.

### Gap 2 — manipulated: 5639 on disk → 5323 processed

```
  Celeb-synthesis on disk              5639      <-- official release count, matches exactly
  Celeb-synthesis retained             5323
  manipulated EXCLUDED                  316   (5.6%)
```
The manuscript reports 5323 manipulated both as input and as processed, implying **zero** exclusions.
The measured figure is **316**.

### Gap 3 — the 875 / 5612 "attempted" figures in the legacy audit are also wrong

`LOCKED_NUMBERS.md` §M4 reports Celeb-DF extraction *attempted* = 875 real / 5612 fake. Neither
matches disk (890 / 5639). Cause found in `exp_m4_missingness.py:66`:

```python
mani = pd.read_csv("data_xception/manifest_celebdf.csv").drop_duplicates("video")
```

The denominator was taken from the **Xception crop manifest** — a *downstream* artifact listing only
videos for which Xception face crops were successfully produced (6487 = 875 + 5612). Using a
downstream product as an upstream denominator understates attempts by 15 real and 27 fake, and
correspondingly inflates the reported success rates (0.912 / 0.9485 instead of the true
0.897 / 0.944). This is a second, independent counting error.

### Consolidated

| Quantity | Manuscript | Measured on disk | Δ |
|---|---|---|---|
| authentic input | 806 | **890** | −84 |
| authentic processed | 798 | 798 | ✓ |
| authentic excluded | 8 | **92** | −84 |
| manipulated input | 5323 | **5639** | −316 |
| manipulated processed | 5323 | 5323 | ✓ |
| manipulated excluded | 0 | **316** | −316 |
| **total input** | **6129** | **6529** | **−400** |
| **total evaluated** | **6121** | **6121** | ✓ |
| **total excluded** | **8** | **408** | **−400** |

Both *processed* counts are correct. Both *input* counts are wrong, and the excluded total is
understated 51-fold.

## 4. Replacement Table 6

Describes the evaluated subset accurately and does not call it the complete dataset. Per-reason
exclusion counts are supplied by the Phase 1.1 gating scan (`manifest/master_manifest.csv`) and are
filled in at Phase 2 rather than estimated here.

> **Table 6.** Celeb-DF v2 evaluation population. PRISM is applied zero-shot to the Celeb-DF v2
> release in full; the official 518-video test list is **not** used, so results are not directly
> comparable with work reporting on that split. Counts are videos.

| Subset | In release | Evaluated | Excluded | Excluded % |
|---|---|---|---|---|
| Celeb-real (authentic) | 590 | 545 | 45 | 7.6% |
| YouTube-real (authentic) | 300 | 253 | 47 | 15.7% |
| **Authentic total** | **890** | **798** | **92** | **10.3%** |
| Celeb-synthesis (manipulated) | 5639 | 5323 | 316 | 5.6% |
| **Total** | **6529** | **6121** | **408** | **6.2%** |

> Exclusions arise from the frozen extractor's gating criteria (fewer than 10 decodable frames; a
> MediaPipe face-detection rate below 50%; fewer than three usable spatial sample frames) and are
> broken down by reason in Table [attrition]. Authentic videos are excluded at nearly twice the rate
> of manipulated ones (10.3% vs 5.6%); the class-dependence of this attrition is tested in
> §[attrition] and must be read alongside the reported real-class recall.

## 5. Consequences and required manuscript changes

1. **Replace 806 with 890** as the authentic input population, and **5323 with 5639** as the
   manipulated input. Retire 806 entirely — it belongs to a pipeline the paper has already disowned.
2. **Report 408 exclusions, not 8.**
3. **State explicitly that the official test list was not used** and that the evaluation covers the
   full release. This is material to comparability with prior Celeb-DF work.
4. **State that YouTube-real is included** in the authentic class.
5. **Flag the class-dependent attrition** (authentic 10.3% vs manipulated 5.6%) — it is a
   selection-bias caveat on the real-recall figure of 0.3972, which is already the weakest number in
   Table 13. Fisher's exact test is run in Phase 2.
6. **Correct `LOCKED_NUMBERS.md` §M4** success rates: the denominators were downstream artifacts.

**No re-run of the zero-shot evaluation is required.** The evaluated population (6121) is correct,
identified, fully enumerated, and reproduces to published precision. Only its *description* changes.
The brief's contingency — "propose re-running the zero-shot evaluation on a defensible, documented
subset, flagging that this would move the 0.6322" — is **not triggered**: the counts reconcile
completely from available evidence, so 0.6322 stands.
