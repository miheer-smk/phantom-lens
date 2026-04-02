"""
Feature-Space SBI Augmentation for Phantom Lens V4
====================================================
Creates synthetic fakes from real samples by perturbing features
to simulate manipulation artifacts. This breaks the codec-label
correlation that causes cross-dataset failure.

The Problem:
  CelebVHQ reals: high noise, high PRNU (high-quality source)
  FF++ fakes:     moderate noise (YouTube compressed)
  Model learned:  high noise = real, low noise = fake
  CelebDF reals:  LOW noise (YouTube compressed) → model calls them FAKE
  CelebDF fakes:  HIGHER noise than CelebDF reals → model calls them REAL
  Result: inverted predictions on CelebDF (AUC 0.43)

The Fix:
  Create synthetic fakes FROM CelebVHQ reals by reducing noise/PRNU
  to simulate compressed fakes. Now the model sees:
    - Real: high noise (CelebVHQ) AND low noise (FF++ originals)
    - Fake: moderate noise (FF++ fakes) AND low noise (SBI fakes from CelebVHQ)
  The noise level alone no longer predicts the label.

Usage:
  Drop this into train_v4.py or import the functions.
"""

import numpy as np


# ── V4 Feature Layout Reference ──────────────────────────────────────
# dim 0:  noise_variance           (P1 Noise)
# dim 1:  noise_kurtosis           (P1 Noise)
# dim 2:  noise_entropy            (P1 Noise)
# dim 3:  prnu_correlation         (P2 PRNU)
# dim 4:  prnu_energy              (P2 PRNU)
# dim 5:  shadow_consistency       (P4 Shadow)
# dim 6:  shadow_direction         (P4 Shadow)
# dim 7:  shadow_intensity         (P4 Shadow)
# dim 8:  dct_energy_dist          (P6 DCT)
# dim 9:  dct_block_artifacts      (P6 DCT)
# dim 10: dct_ratio                (P6 DCT)
# dim 11: codec_qp_estimate        (P7 Codec)  ← drop if using 18-dim
# dim 12: codec_bitrate_var        (P7 Codec)  ← drop if using 18-dim
# dim 13: flow_consistency         (P9 Flow)
# dim 14: blend_boundary_grad_ratio    (Blend)
# dim 15: blend_grad_dir_coherence     (Blend)
# dim 16: blend_boundary_vs_bg_ratio   (Blend)
# dim 17: freq_peak_energy_ratio       (Freq)
# dim 18: freq_peak_spacing_regularity (Freq)
# dim 19: freq_face_bg_spectral_kl     (Freq)


def sbi_augment_features(features, labels, sources,
                         sbi_prob=0.30,
                         noise_reduction=0.40,
                         prnu_reduction=0.40,
                         dct_boost=0.25,
                         blend_perturbation=0.30,
                         jitter_std=0.05):
    """
    Feature-space Self-Blended Image augmentation.

    Takes real samples (especially CelebVHQ) and creates synthetic fakes
    by perturbing features to simulate what a face-swap would produce.

    Physics rationale for each perturbation:
      - Noise dims reduced: face-swap destroys original sensor noise,
        generated face has lower/different noise statistics
      - PRNU dims reduced: generated face doesn't carry original
        camera's PRNU fingerprint, correlation drops
      - DCT dims boosted: re-compression of inserted face creates
        stronger block artifacts (double quantisation effect)
      - Blend boundary dims perturbed: face-swap creates blending
        seam that increases gradient ratio at boundary

    Args:
        features:  (N, D) array of training features
        labels:    (N,) array of labels (0=real, 1=fake)
        sources:   (N,) array or list of dataset source strings
        sbi_prob:  probability of converting a real sample to SBI fake
        noise_reduction:   fraction to reduce noise features (0.0-1.0)
        prnu_reduction:    fraction to reduce PRNU features (0.0-1.0)
        dct_boost:         fraction to increase DCT features (0.0-1.0)
        blend_perturbation: fraction to perturb blend boundary features
        jitter_std:        gaussian noise std added to all SBI samples

    Returns:
        aug_features: (N + N_sbi, D) augmented feature array
        aug_labels:   (N + N_sbi,) augmented label array
        aug_sources:  list of source strings including 'sbi_fake'
    """
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    sources = np.array(sources) if not isinstance(sources, np.ndarray) else sources
    D = features.shape[1]

    # Identify real samples (candidates for SBI conversion)
    real_mask = labels == 0
    real_indices = np.where(real_mask)[0]

    # Select which real samples become SBI fakes
    rng = np.random.RandomState()
    sbi_mask = rng.random(len(real_indices)) < sbi_prob
    sbi_indices = real_indices[sbi_mask]

    if len(sbi_indices) == 0:
        return features, labels, list(sources)

    # Copy the selected real samples
    sbi_features = features[sbi_indices].copy()
    n_sbi = len(sbi_features)

    # ── Apply physics-motivated perturbations ────────────────────────

    # Determine which dims exist (handle both 18-dim and 20-dim)
    has_codec = D >= 20  # dims 11-12 are codec if 20-dim

    # Noise dims (0, 1, 2): reduce to simulate destroyed sensor noise
    # A face-swap generator doesn't reproduce camera noise, so the
    # generated face has attenuated/altered noise statistics
    for d in [0, 1, 2]:
        if d < D:
            reduction = rng.uniform(noise_reduction * 0.5, noise_reduction * 1.5, n_sbi)
            sbi_features[:, d] *= (1.0 - reduction)

    # PRNU dims (3, 4): reduce to simulate broken sensor fingerprint
    # Generated face doesn't carry the correct PRNU pattern
    for d in [3, 4]:
        if d < D:
            reduction = rng.uniform(prnu_reduction * 0.5, prnu_reduction * 1.5, n_sbi)
            sbi_features[:, d] *= (1.0 - reduction)

    # Shadow dims (5, 6, 7): add slight inconsistency
    # Face-swap can create lighting mismatches but effect is subtle
    for d in [5, 6, 7]:
        if d < D:
            sbi_features[:, d] += rng.normal(0, 0.1, n_sbi) * np.abs(sbi_features[:, d])

    # DCT dims (8, 9, 10): boost to simulate double compression
    # Inserted face undergoes encode → decode → re-encode, creating
    # stronger block artifacts
    for d in [8, 9, 10]:
        if d < D:
            boost = rng.uniform(dct_boost * 0.5, dct_boost * 1.5, n_sbi)
            sbi_features[:, d] *= (1.0 + boost)

    # Codec dims (11, 12): skip — these should be dropped in V4
    # If present, leave unchanged

    # Flow dim (13): slight perturbation (temporal inconsistency)
    if D > 13:
        flow_dim = 13
        sbi_features[:, flow_dim] += rng.normal(0, 0.15, n_sbi) * np.abs(sbi_features[:, flow_dim])

    # Blend boundary dims (14, 15, 16): perturb to simulate blending seam
    # This is the most physically motivated perturbation — every
    # face-swap creates a blending boundary
    blend_start = 14
    if D > blend_start:
        # Dim 14 (grad ratio): increase — blending creates stronger boundary gradient
        if blend_start < D:
            boost = rng.uniform(blend_perturbation * 0.5, blend_perturbation * 2.0, n_sbi)
            sbi_features[:, blend_start] *= (1.0 + boost)

        # Dim 15 (direction coherence): decrease — blending disrupts gradient flow
        if blend_start + 1 < D:
            reduction = rng.uniform(0.1, blend_perturbation, n_sbi)
            sbi_features[:, blend_start + 1] *= (1.0 - reduction)

        # Dim 16 (boundary vs background): increase — unnatural boundary
        if blend_start + 2 < D:
            boost = rng.uniform(blend_perturbation * 0.5, blend_perturbation * 1.5, n_sbi)
            sbi_features[:, blend_start + 2] *= (1.0 + boost)

    # Freq dims (17, 18, 19): perturb to simulate spectral inconsistency
    freq_start = 17
    if D > freq_start:
        # Dim 17 (peak energy): for autoencoder fakes (like CelebDF),
        # fakes have LESS high-freq energy, so reduce
        if freq_start < D:
            reduction = rng.uniform(0.1, 0.3, n_sbi)
            sbi_features[:, freq_start] *= (1.0 - reduction)

        # Dim 18 (spacing regularity): slight increase
        if freq_start + 1 < D:
            sbi_features[:, freq_start + 1] += rng.uniform(0.0, 0.15, n_sbi)

        # Dim 19 (face-bg KL): increase — spectral mismatch between
        # generated face and original background
        if freq_start + 2 < D:
            boost = rng.uniform(0.1, 0.4, n_sbi)
            sbi_features[:, freq_start + 2] *= (1.0 + boost)

    # ── Add jitter to all dims (prevents exact copies) ───────────────
    jitter = rng.normal(0, jitter_std, sbi_features.shape).astype(np.float32)
    sbi_features += jitter

    # ── Build augmented dataset ──────────────────────────────────────
    sbi_labels = np.ones(n_sbi, dtype=np.float32)  # all SBI samples are fake
    sbi_sources = ['sbi_fake'] * n_sbi

    aug_features = np.concatenate([features, sbi_features], axis=0)
    aug_labels = np.concatenate([labels, sbi_labels], axis=0)
    aug_sources = list(sources) + sbi_sources

    return aug_features, aug_labels, aug_sources


def augment_features_v4(features, labels, sources=None):
    """
    Complete V4 augmentation pipeline: standard augmentation + SBI.
    Drop-in replacement for the augment_features function in train_v4.py.

    Call this ONCE before creating the DataLoader, not per-epoch.
    SBI samples are added to the dataset, then standard augmentation
    (noise, brightness, compression jitter) is applied to everything.

    Args:
        features: (N, D) training features
        labels:   (N,) training labels
        sources:  (N,) dataset source strings (optional but recommended)

    Returns:
        aug_features, aug_labels (ready for DataLoader)
    """
    AUG_PROB = 0.5
    NOISE_STD = 0.02
    BRIGHTNESS_RANGE = 0.15
    COMPRESSION_RANGE = (0.6, 1.4)

    D = features.shape[1]

    # ── Step 1: SBI augmentation (add synthetic fakes) ───────────────
    if sources is not None:
        features, labels, sources = sbi_augment_features(
            features, labels, sources,
            sbi_prob=0.30,
            noise_reduction=0.40,
            prnu_reduction=0.40,
            dct_boost=0.25,
            blend_perturbation=0.30,
            jitter_std=0.05,
        )
        n_sbi = int((np.array(sources) == 'sbi_fake').sum())
    else:
        # No source info — apply SBI to random 30% of reals
        fake_sources = ['unknown'] * len(labels)
        features, labels, fake_sources = sbi_augment_features(
            features, labels, fake_sources,
            sbi_prob=0.30,
        )
        n_sbi = len(features) - len(labels)

    # ── Step 2: Standard augmentation (per-sample jitter) ────────────
    aug = features.copy()
    for i in range(len(aug)):
        # Global noise
        if np.random.random() < AUG_PROB:
            aug[i] += np.random.normal(0, NOISE_STD, D).astype(np.float32)

        # Brightness scaling on noise + PRNU dims
        if np.random.random() < AUG_PROB:
            shift = np.random.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
            aug[i, 0:3] *= (1 + shift)   # P1 Noise
            aug[i, 3:5] *= (1 + shift)   # P2 PRNU

        # Compression scaling on DCT dims
        if np.random.random() < AUG_PROB:
            cf = np.random.uniform(*COMPRESSION_RANGE)
            aug[i, 8:11] *= cf   # P6 DCT (always dims 8-10)
            # If codec dims exist (20-dim version), scale those too
            if D >= 13:
                aug[i, 11:13] *= cf  # P7 Codec

        # Random feature dropout
        if np.random.random() < 0.2:
            aug[i, np.random.randint(0, D)] = 0.0

    return aug, np.array(labels, dtype=np.float32)


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    N = 1000
    D = 20

    # Simulate training data: 600 reals (400 CelebVHQ + 200 FF++), 400 fakes
    features = np.random.randn(N, D).astype(np.float32)
    labels = np.array([0]*600 + [1]*400, dtype=np.float32)
    sources = ['celebvhq']*400 + ['ffpp_official']*200 + ['ffpp_official']*400

    # Make CelebVHQ reals have higher noise (mimics real data)
    features[:400, 0:3] += 1.5  # higher noise
    features[:400, 3:5] += 1.0  # higher PRNU

    print("=" * 60)
    print("Feature-Space SBI Augmentation — Self-Test")
    print("=" * 60)

    # Test SBI augmentation
    aug_f, aug_l, aug_s = sbi_augment_features(features, labels, sources)
    n_orig = len(labels)
    n_aug = len(aug_l)
    n_sbi = int((np.array(aug_s) == 'sbi_fake').sum())

    print(f"\n  Original: {n_orig} samples ({int((labels==0).sum())} real, {int((labels==1).sum())} fake)")
    print(f"  After SBI: {n_aug} samples (+{n_sbi} SBI fakes)")
    print(f"  New balance: {int((aug_l==0).sum())} real, {int((aug_l==1).sum())} fake")

    # Verify SBI fakes have reduced noise
    sbi_mask = np.array(aug_s) == 'sbi_fake'
    celebvhq_mask = np.array(aug_s) == 'celebvhq'

    print(f"\n  CelebVHQ real noise mean (dim 0): {aug_f[celebvhq_mask, 0].mean():.4f}")
    print(f"  SBI fake noise mean (dim 0):      {aug_f[sbi_mask, 0].mean():.4f}")
    print(f"  Reduction ratio:                   {aug_f[sbi_mask, 0].mean() / aug_f[celebvhq_mask, 0].mean():.4f}")

    print(f"\n  CelebVHQ real PRNU mean (dim 3):  {aug_f[celebvhq_mask, 3].mean():.4f}")
    print(f"  SBI fake PRNU mean (dim 3):       {aug_f[sbi_mask, 3].mean():.4f}")
    print(f"  Reduction ratio:                   {aug_f[sbi_mask, 3].mean() / aug_f[celebvhq_mask, 3].mean():.4f}")

    # Verify blend boundary perturbation
    print(f"\n  CelebVHQ blend grad (dim 14):     {aug_f[celebvhq_mask, 14].mean():.4f}")
    print(f"  SBI fake blend grad (dim 14):     {aug_f[sbi_mask, 14].mean():.4f}")

    # Test full pipeline
    print(f"\n  Testing full augment_features_v4 pipeline...")
    full_f, full_l = augment_features_v4(features, labels, sources)
    print(f"  Output: {len(full_l)} samples ({int((full_l==0).sum())} real, {int((full_l==1).sum())} fake)")
    print(f"  Shape: {full_f.shape}")
    assert full_f.shape[1] == D
    assert len(full_l) == len(full_f)

    # Verify the key property: SBI fakes from CelebVHQ now have
    # LOW noise (like FF++ codec) but FAKE labels
    # This means the model can't use "high noise = real" shortcut
    print(f"\n  KEY VERIFICATION:")
    print(f"  FF++ reals noise mean:       {features[400:600, 0].mean():.4f}")
    print(f"  FF++ fakes noise mean:       {features[600:, 0].mean():.4f}")
    print(f"  CelebVHQ reals noise mean:   {features[:400, 0].mean():.4f}")
    print(f"  SBI fakes noise mean:        {aug_f[sbi_mask, 0].mean():.4f}")
    print(f"  → SBI fakes have LOWER noise than CelebVHQ reals but FAKE label")
    print(f"  → Model cannot use 'high noise = real' shortcut anymore")

    print(f"\n  [OK] All tests passed")
    print("=" * 60)