# Exclusions: why videos drop out, measured

Every reported population is smaller than the corpus it came from. This file states why, using a
controlled vocabulary, with the mechanism **observed** rather than inferred.

## Controlled vocabulary

| Reason | Meaning |
|---|---|
| `file_missing` | the listed file is not on disk |
| `decode_failure` | present but not decodable |
| `mediapipe_no_face` | face detected in < 50% of sampled frames |
| `insufficient_valid_frames` | fewer than 10 decodable frames, or fewer than 3 usable spatial samples |
| `feature_computation_failure` | passed every gate, but a descriptor raised during computation |
| `retained` | included in the evaluation |

**`file_missing` and `decode_failure` are zero for every corpus reported.** All attrition is one of
the other three.

## Measured attrition

| Corpus | input | retained | mediapipe_no_face | insufficient_frames | computation_failure |
|---|---|---|---|---|---|
| FF++ (10 subsets) | 10,000 | 9,572 | 188 | 0 | 240 |
| Celeb-DF v2 | 6,529 | 6,121 | 54 | 1 | 353 |
| WildDeepfake | 174 | 124 | 5 | 3 | 42 |
| DF40 (5 methods) | 3,697 | 3,664 | 31 | 2 | 0 |

## The computation-failure bucket, named

An instrumented extractor with per-descriptor exception capture was run over the 635
computation-failure videos:

| Failing descriptor | videos | share |
|---|---|---|
| **`t5_codec_residual`** (`extract_codec_temporal_residual`) | **540** | **85%** |
| none — recomputed cleanly on re-run | 95 | 15% |
| all 14 other descriptor blocks | 0 | 0% |

All 540 raise the same exception: `np.histogram` on an all-NaN residual array. Because
`process_single_video` wraps the whole computation in one `try/except`, **a single bad descriptor of
fifty voids the entire video.**

Failure rate by corpus: FF++ **2.4%**, Celeb-DF **4.0%**, WildDeepfake **24.1%**. The tenfold
elevation on pre-cropped frames reflects the absent codec substrate; the 2.4% on ordinary video
shows this is a latent numerical fragility, not only a substrate effect.

## The shipped guard — an improvement over the version used for the reported results

`src/preprocessing/precompute_features_seeded.py` can guard the histogram against non-finite input
and capture exceptions per descriptor, so one bad descriptor yields **that descriptor as missing**
rather than voiding the video.

**It is off by default.** Enabling it would retain **540 of the 635 currently-excluded videos**
(FF++ +240, Celeb-DF +258, WildDeepfake +42), each carrying one missing descriptor of fifty. That is
a materially different evaluation population from the one behind the published results, so the
default preserves the reported behaviour and the improvement is opt-in.

## Not fully reproducible

95 Celeb-DF exclusions recompute cleanly on re-run: the evaluated population is reproducible to
roughly **±1.5%** on that corpus (n would move 6,121 → ~6,216). The likely path is MediaPipe landmark
jitter altering the face mask and hence the residual. Enabling the guard removes this sensitivity,
since a NaN descriptor would no longer void a video.
