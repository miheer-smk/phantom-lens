# Ground-truth 50-feature → pillar-group mapping

Source: `src/precompute_features_best.py` header (lines 9–33) + `FEATURE_NAMES_SPATIAL` /
`FEATURE_NAMES_TEMPORAL`. This is the authoritative grouping for Experiment 1 and 3.

## 20-pillar (fine) grouping

| pillar_id | group_name | n | features |
|---|---|---|---|
| P1 | noise_physics | 3 | s_noise_vmr, s_noise_res_std, s_noise_hf_ratio |
| P2 | prnu_camera | 2 | s_prnu_energy, s_prnu_face_periph |
| P4 | shadow_light | 2 | s_shadow_score, s_face_bg_diff |
| P6 | compression_forensics | 3 | s_benford_dev, s_block_artifact, s_dbl_compress |
| P8 | motion_blur_spatial | 1 | s_blur_mag |
| P9 | optical_flow | 2 | s_flow_mag, s_flow_dir_consist |
| T1 | temporal_noise_stability | 3 | t_noise_temporal_corr, t_noise_corr_std, t_noise_spectral_entropy |
| T2 | rppg_cardiac | 4 | t_rppg_snr, t_rppg_peak_prominence, t_rppg_interregion_corr, t_rppg_harmonic_ratio |
| T3 | temporal_prnu | 2 | t_prnu_temporal_stability, t_prnu_face_vs_bg |
| T4 | face_structural_stability | 3 | t_face_ssim_mean, t_face_ssim_std, t_face_ssim_min |
| T5 | codec_temporal_residual | 2 | t_residual_flow_corr, t_residual_entropy |
| T6 | landmark_trajectory | 4 | t_landmark_jitter, t_landmark_accel_var, t_landmark_velocity_autocorr, t_jaw_chin_rigidity |
| T7 | rigid_geometry | 3 | t_rigid_dist_var, t_interpupillary_std, t_nose_bridge_std |
| T8 | face_bg_edge_coherence | 3 | t_boundary_grad_temporal, t_boundary_color_disc, t_boundary_freq_leakage |
| T9 | skin_texture_coherence | 2 | t_skin_texture_corr, t_texture_warp_residual |
| T10 | color_transfer | 2 | t_skin_color_jitter, t_skin_bg_decorrelation |
| T11 | specular_temporal | 2 | t_specular_stability, t_specular_symmetry |
| T12 | blink_dynamics | 3 | t_blink_rate, t_blink_duration, t_blink_symmetry |
| T13 | motion_blur_coupling | 2 | t_motion_blur_coupling, t_coupling_consistency |
| T14 | dct_temporal | 2 | t_dct_temporal_std, t_dct_temporal_autocorr |

Total = 13 spatial + 37 temporal = **50** ✓

## Coarse grouping matching the reviewer response's example table

The reviewer example named: Noise physics (3), PRNU-inspired residual (2), rPPG (4),
Landmark geometry (7), Blink dynamics (3). Mapping onto the pillars above:

| reviewer_group | n | composed of pillars |
|---|---|---|
| Noise physics | 3 | P1 |
| PRNU-inspired residual | 2 | P2 (spatial only) — NB: T3 temporal PRNU (2) may or may not belong here — **A1** |
| rPPG | 4 | T2 |
| Landmark geometry | 7 | T6 (4) + T7 (3) |
| Blink dynamics | 3 | T12 |

The reviewer's 5 groups cover only 19 of 50 features (it was an explicitly truncated
example). A complete coarse scheme covering all 50 is NOT specified by the reviewers →
see open_ambiguities.md #A1 for the exact granularity decision Experiment 1 needs.
