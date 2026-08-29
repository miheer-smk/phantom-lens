# Dataset access and placement

**No dataset video is redistributed in this archive.** Each corpus must be obtained from its
owners under their own terms. Place them as below; the code reads these paths and nothing else.

```
<DATA_ROOT>/
  ffpp/
    original_sequences/youtube/{c23,c40}/videos/*.mp4
    manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/{c23,c40}/videos/*.mp4
  celebdf_v2/
    Celeb-real/*.mp4
    YouTube-real/*.mp4
    Celeb-synthesis/*.mp4
    List_of_testing_videos.txt
  wilddeepfake/
    test/{real,fake}/*.png            # distributed as 224x224 face crops, NOT videos
  df40/
    <method>/<method>/{ff,cdf}/*.mp4  # inswap, facedancer, wav2lip, sadtalker, hyperreenact
```

| Corpus | Obtain from | Notes |
|---|---|---|
| FaceForensics++ | https://github.com/ondyari/FaceForensics — access form | c23 and c40. The official 720/140/140 identity split ships here in `splits/`. |
| Celeb-DF v2 | https://github.com/yuezunli/celeb-deepfakeforensics — access form | **All three folders**, including YouTube-real. See the note below on the evaluation population. |
| WildDeepfake | https://github.com/OpenTAI/wild-deepfake — access form | Distributed as pre-cropped face frames. **Outside PRISM's operating domain** — run `check_substrate()` first and read the paper's operating-domain section. |
| DF40 | https://github.com/YZY-stack/DF40 — access form | Use the **video** variants, not the frame variants. |

## Celeb-DF v2 evaluation population — read before comparing

The paper evaluates zero-shot on the **entire Celeb-DF v2 release**, not the official 518-video
test list. Counts, so results are comparable:

| Subset | in release | evaluated | excluded |
|---|---|---|---|
| Celeb-real | 590 | 545 | 45 |
| YouTube-real | 300 | 253 | 47 |
| **authentic total** | **890** | **798** | **92** |
| Celeb-synthesis | 5639 | 5323 | 316 |
| **total** | **6529** | **6121** | **408** |

Exclusions arise from the extractor's gates and from a descriptor-level computation failure; see
`docs/EXCLUSIONS.md`. **Authentic videos are excluded at nearly twice the rate of manipulated ones
(10.3% vs 5.6%, Fisher OR 0.515, p = 1.8 × 10⁻⁶)** — a selection bias that must be read alongside
the reported real-class recall.

## Reproducing from released features instead

Every table in the paper can be reproduced from `results/` without any dataset, using the released
per-video scores and feature matrices. Re-extraction from raw video is only needed to regenerate
the feature matrices themselves — and see the reproducibility caveats in the README before doing so.
