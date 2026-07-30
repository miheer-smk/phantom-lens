# Sealed evaluation — provenance & integrity note (Celeb-DF-v2 test)

**Result: Celeb-DF-v2 sealed TEST AUC = 0.7133, 95% CI [0.687, 0.746] (n=2273, 372 real / 1901 fake).**
Frozen model: 196-D E1-expanded + RF+ExtraTrees+LGBM_d6 rank ensemble, trained on FF++ train only (seed 42).

## Why the unseal log shows TWO entries (must be disclosed)
The audit log (`sealed.py`) records two `UNSEAL celebdf_test` timestamps:
- **15:08:19** — first invocation. It computed the celebdf_test AUC in memory but then **crashed in the
  FF++-test reporting branch** (`ff[partition=="test"]` was empty — FF++ test features are not in
  `plain_everyone_E3`, which is train+val only). The crash occurred **before any result was printed or saved**,
  so **no test metric was observed or used**.
- **17:00:06** — second invocation, after fixing the FF++ branch to be optional. Captured the result.

## Why this is ONE evaluation of a frozen model, not test-based tuning
1. The model, representation, and all hyperparameters were **frozen and committed (`c1a77bb`) before either
   unseal** — see `trackE_FREEZE.md`. Nothing about the model changed between the two invocations.
2. The only code change between invocations was in the **FF++-test reporting/saving branch**, which does not
   touch celebdf_test scoring at all.
3. celebdf_test scoring is **deterministic** (seed 42, fixed features): the second run reproduces the exact
   number the first (crashed) run had computed. No information from the first run influenced the model.
4. **No celebdf_test metric was observed between the two runs** (the first crashed before emitting anything),
   so there was no opportunity for test-guided decisions.

**Conclusion:** this constitutes a single, honest evaluation of a pre-committed frozen model. The two log lines
are a crash-and-recapture of the identical deterministic computation, not two independent peeks. Methods should
state this plainly rather than claim a single clean log line.

## Predicted vs actual (pre-registered, trackE_preregistration.md)
- Predicted: **0.68**, 80% interval **[0.65, 0.71]** (written before unseal).
- Actual: **0.7133** — above the pre-registered interval. The prediction was too conservative; the
  identity-grouped dev CV (0.7125) proved well-calibrated to held-out identities (test 0.7133 ≈ dev CV).
- Single-model reference on test: ExtraTrees 0.7143, RF 0.7059, LGBM 0.691 — the ensemble/ExtraTrees choice
  was not outcome-determining (as pre-stated in the freeze document).

## FF++ test (in-distribution) — NOT scored in this run
FF++ test 196-D features were not extracted (not needed for the cross-dataset question). The in-distribution
number is established separately on `major-revision`. If an FF++-test number on the frozen model is wanted, it
requires extracting FF++ test-identity features; it does not affect the sealed cross-dataset result above.

## Methods disclosure (for the paper)
57 celebdf_dev evaluations preceded the sealed test; model+representation fixed a priori (freeze doc); the sealed
Celeb-DF test half (27 identities / 2273 videos, identity-disjoint from dev) was evaluated once (see the crash-
and-recapture note above). Sealed cross-dataset AUC = 0.7133 [0.687, 0.746].
