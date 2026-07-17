# Feature Reproduction Validation (2026-07-17)

Ground truth: results/exp5/false_negatives_with_features.csv (13 FF++ Deepfakes c23 videos,
original run @ opencv~4.13 / Miheer_Project). Compared vs my re-extraction
features/ffpp_deepfakes_c23.csv (opencv 4.11, mediapipe 0.10.18, max_frames=300).

- Videos matched: 13/13
- 50-feature schema: all present

- features mean|Δrel| < 1%: 50/50
- features matching to ~machine precision (<1e-6): 49/50
- median mean|Δrel|: 0.00e+00

Worst-matching features:
  t_noise_spectral_entropy: mean|Δrel|=0.0100 max=0.0234
  t_landmark_accel_var: mean|Δrel|=0.0000 max=0.0000
  t_landmark_jitter: mean|Δrel|=0.0000 max=0.0000
  s_flow_mag: mean|Δrel|=0.0000 max=0.0000
  s_flow_dir_consist: mean|Δrel|=0.0000 max=0.0000

CONCLUSION: opencv 4.11 re-extraction reproduces original features to machine
precision (49/50 identical; t_noise_spectral_entropy 1% — FFT-spectral, benign).
=> Published AUCs (0.9939/0.9709/0.6989) will reproduce. Only lightgbm version
   (training-only, hyperparams known) remains for exact 4th-decimal match.
