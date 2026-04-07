#!/usr/bin/env python3
"""
PHANTOM LENS V2 — Feature Space Evaluation
=============================================
Evaluates physics feature quality using clustering and distribution metrics.

Metrics generated:
  1. Silhouette Score — cluster quality (-1 to 1, higher = better)
  2. Davies-Bouldin Index (DBI) — cluster separation (lower = better)
  3. Mahalanobis Distance Analysis — distribution separation
  4. Per-feature Cohen's d heatmap
  5. t-SNE visualization (train vs test domain)
  6. Feature correlation matrix
  7. Per-pillar contribution analysis

Usage:
    python evaluate_v3.py \
        --train_csv features_ffpp.csv \
        --test_csv features_celebdf.csv \
        --output_dir evaluation_v3/
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    silhouette_samples,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare(csv_paths):
    """Load feature CSVs and return X, y, feature_names, df."""
    dfs = []
    for p in csv_paths:
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    
    feature_cols = sorted([c for c in df.columns
                           if c.startswith("s_") or c.startswith("t_")])
    
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())
    
    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values.astype(int)
    return X, y, feature_cols, df


# ============================================================================
# 1. SILHOUETTE SCORE
# ============================================================================

def compute_silhouette(X, y, output_dir):
    """Compute and visualize Silhouette Score."""
    print("\n" + "=" * 60)
    print("1. SILHOUETTE SCORE ANALYSIS")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Overall silhouette score
    sil_score = silhouette_score(X_scaled, y, metric="euclidean")
    print(f"  Overall Silhouette Score: {sil_score:.4f}")
    print(f"  Interpretation: ", end="")
    if sil_score > 0.5:
        print("Strong cluster separation")
    elif sil_score > 0.25:
        print("Moderate cluster structure")
    elif sil_score > 0.0:
        print("Weak but present cluster structure")
    else:
        print("No meaningful cluster structure")
    
    # Per-sample silhouette values
    sil_samples = silhouette_samples(X_scaled, y, metric="euclidean")
    
    # Silhouette plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    y_lower = 10
    colors = ["#378ADD", "#E24B4A"]
    class_names = ["Real", "Fake"]
    
    for i, label in enumerate([0, 1]):
        cluster_sil = sil_samples[y == label]
        cluster_sil.sort()
        
        size = len(cluster_sil)
        y_upper = y_lower + size
        
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_sil,
                          alpha=0.7, color=colors[i],
                          label=f"{class_names[i]} (n={size})")
        
        ax.text(-0.05, y_lower + 0.5 * size, class_names[i],
                fontsize=11, fontweight="bold", va="center")
        y_lower = y_upper + 10
    
    ax.axvline(x=sil_score, color="black", linestyle="--", lw=1.5,
               label=f"Mean = {sil_score:.4f}")
    ax.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax.set_ylabel("Samples", fontsize=12)
    ax.set_title("Silhouette Analysis — Real vs Fake Feature Space", fontsize=13)
    ax.legend(loc="upper right")
    ax.set_xlim([-0.3, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "silhouette_plot.png"), dpi=200)
    plt.close()
    
    # Per-class silhouette
    real_sil = sil_samples[y == 0].mean()
    fake_sil = sil_samples[y == 1].mean()
    print(f"  Real class silhouette: {real_sil:.4f}")
    print(f"  Fake class silhouette: {fake_sil:.4f}")
    
    # Misclassification proxy: samples with negative silhouette
    neg_real = (sil_samples[y == 0] < 0).sum()
    neg_fake = (sil_samples[y == 1] < 0).sum()
    print(f"  Negative silhouette (real): {neg_real}/{(y==0).sum()} "
          f"({neg_real/(y==0).sum()*100:.1f}%)")
    print(f"  Negative silhouette (fake): {neg_fake}/{(y==1).sum()} "
          f"({neg_fake/(y==1).sum()*100:.1f}%)")
    
    return {
        "silhouette_score": float(sil_score),
        "silhouette_real": float(real_sil),
        "silhouette_fake": float(fake_sil),
        "negative_sil_real_pct": float(neg_real / (y == 0).sum() * 100),
        "negative_sil_fake_pct": float(neg_fake / (y == 1).sum() * 100),
    }


# ============================================================================
# 2. DAVIES-BOULDIN INDEX (DBI)
# ============================================================================

def compute_dbi(X, y, output_dir):
    """Compute Davies-Bouldin Index."""
    print("\n" + "=" * 60)
    print("2. DAVIES-BOULDIN INDEX (DBI)")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dbi = davies_bouldin_score(X_scaled, y)
    print(f"  DBI Score: {dbi:.4f}")
    print(f"  Interpretation: ", end="")
    if dbi < 0.5:
        print("Excellent separation (well-defined clusters)")
    elif dbi < 1.0:
        print("Good separation")
    elif dbi < 1.5:
        print("Moderate separation")
    elif dbi < 2.0:
        print("Weak separation")
    else:
        print("Poor separation (overlapping clusters)")
    
    # Compute per-feature DBI to identify strongest individual features
    per_feature_dbi = {}
    for i in range(X_scaled.shape[1]):
        try:
            X_single = X_scaled[:, i].reshape(-1, 1)
            # DBI needs at least 2 features or repeated singleton.
            # For single feature we use a proxy: add a noise dimension
            X_aug = np.column_stack([X_single, np.random.randn(len(X_single)) * 0.01])
            d = davies_bouldin_score(X_aug, y)
            per_feature_dbi[i] = d
        except Exception:
            per_feature_dbi[i] = 99.0
    
    return {
        "dbi_score": float(dbi),
    }


# ============================================================================
# 3. MAHALANOBIS DISTANCE ANALYSIS
# ============================================================================

def compute_mahalanobis(X, y, feature_names, output_dir):
    """Compute Mahalanobis distance between real and fake distributions."""
    print("\n" + "=" * 60)
    print("3. MAHALANOBIS DISTANCE ANALYSIS")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    real_data = X_scaled[y == 0]
    fake_data = X_scaled[y == 1]
    
    # Mean vectors
    mu_real = real_data.mean(axis=0)
    mu_fake = fake_data.mean(axis=0)
    
    # Pooled covariance matrix
    n_real, n_fake = len(real_data), len(fake_data)
    cov_real = np.cov(real_data, rowvar=False)
    cov_fake = np.cov(fake_data, rowvar=False)
    cov_pooled = ((n_real - 1) * cov_real + (n_fake - 1) * cov_fake) / (n_real + n_fake - 2)
    
    # Regularize to avoid singularity
    cov_pooled += np.eye(cov_pooled.shape[0]) * 1e-6
    
    try:
        cov_inv = np.linalg.inv(cov_pooled)
        maha_dist = mahalanobis(mu_real, mu_fake, cov_inv)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        cov_inv = np.linalg.pinv(cov_pooled)
        diff = mu_real - mu_fake
        maha_dist = np.sqrt(diff @ cov_inv @ diff)
    
    print(f"  Mahalanobis Distance (full feature space): {maha_dist:.4f}")
    print(f"  Interpretation: ", end="")
    if maha_dist > 3.0:
        print("Excellent separation (distributions are well-separated)")
    elif maha_dist > 2.0:
        print("Good separation")
    elif maha_dist > 1.0:
        print("Moderate separation")
    else:
        print("Weak separation (substantial overlap)")
    
    # Per-feature Mahalanobis (univariate = Cohen's d)
    per_feature_maha = {}
    for i, fname in enumerate(feature_names):
        real_vals = X_scaled[y == 0, i]
        fake_vals = X_scaled[y == 1, i]
        
        mean_diff = abs(real_vals.mean() - fake_vals.mean())
        pooled_var = ((len(real_vals) - 1) * real_vals.var() +
                      (len(fake_vals) - 1) * fake_vals.var()) / \
                     (len(real_vals) + len(fake_vals) - 2)
        d = mean_diff / (np.sqrt(pooled_var) + 1e-8)
        per_feature_maha[fname] = d
    
    # Per-sample Mahalanobis from each class centroid
    try:
        maha_real_from_real = []
        maha_fake_from_real = []
        for x in real_data[:500]:  # subsample for speed
            maha_real_from_real.append(mahalanobis(x, mu_real, cov_inv))
        for x in fake_data[:500]:
            maha_fake_from_real.append(mahalanobis(x, mu_real, cov_inv))
        
        # Plot distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribution of distances from real centroid
        axes[0].hist(maha_real_from_real, bins=50, alpha=0.7, color="#378ADD",
                     label="Real", density=True)
        axes[0].hist(maha_fake_from_real, bins=50, alpha=0.7, color="#E24B4A",
                     label="Fake", density=True)
        axes[0].set_xlabel("Mahalanobis Distance from Real Centroid")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Distance from Real Centroid")
        axes[0].legend()
        
        # Per-feature effect sizes
        sorted_feats = sorted(per_feature_maha.items(),
                              key=lambda x: x[1], reverse=True)
        top_n = min(20, len(sorted_feats))
        names = [x[0] for x in sorted_feats[:top_n]]
        vals = [x[1] for x in sorted_feats[:top_n]]
        colors = ["#e24b4a" if v > 0.8 else "#ef9f27" if v > 0.5
                  else "#378add" for v in vals]
        
        axes[1].barh(range(top_n), vals[::-1], color=colors[::-1])
        axes[1].set_yticks(range(top_n))
        axes[1].set_yticklabels(names[::-1], fontsize=8)
        axes[1].axvline(x=0.8, color="red", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Effect Size (|Cohen's d|)")
        axes[1].set_title("Top 20 Features by Discriminative Power")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mahalanobis_analysis.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not compute per-sample Mahalanobis: {e}")
    
    # Pillar-level analysis
    print(f"\n  --- Per-Pillar Analysis ---")
    pillars = {
        "P1_Noise": [f for f in feature_names if "noise" in f and f.startswith("s_")],
        "P2_PRNU": [f for f in feature_names if "prnu" in f and f.startswith("s_")],
        "P4_Shadow": [f for f in feature_names if "shadow" in f or "face_bg" in f],
        "P6_Compress": [f for f in feature_names if any(k in f for k in ["benford", "block", "dbl"])],
        "P8_Blur": [f for f in feature_names if "blur" in f and f.startswith("s_")],
        "P9_Flow": [f for f in feature_names if "flow" in f and f.startswith("s_")],
        "T1_TempNoise": [f for f in feature_names if "noise" in f and f.startswith("t_")],
        "T2_rPPG": [f for f in feature_names if "rppg" in f],
        "T3_TempPRNU": [f for f in feature_names if "prnu" in f and f.startswith("t_")],
        "T4_FaceSSIM": [f for f in feature_names if "ssim" in f],
        "T5_CodecRes": [f for f in feature_names if "residual" in f],
        "T6_Landmark": [f for f in feature_names if any(k in f for k in ["landmark", "jitter", "accel", "velocity", "jaw"])],
        "T7_RigidGeom": [f for f in feature_names if any(k in f for k in ["rigid", "interpupillary", "nose_bridge"])],
        "T8_Boundary": [f for f in feature_names if "boundary" in f],
        "T9_SkinTex": [f for f in feature_names if any(k in f for k in ["skin_texture", "texture_warp"])],
        "T10_Color": [f for f in feature_names if any(k in f for k in ["skin_color", "skin_bg"])],
        "T11_Specular": [f for f in feature_names if "specular" in f],
        "T12_Blink": [f for f in feature_names if "blink" in f],
        "T13_MotBlur": [f for f in feature_names if "coupling" in f or "motion_blur" in f],
        "T14_DCT": [f for f in feature_names if "dct" in f and f.startswith("t_")],
    }
    
    pillar_scores = {}
    for pillar_name, feats in pillars.items():
        if not feats:
            continue
        d_values = [per_feature_maha.get(f, 0) for f in feats]
        avg_d = np.mean(d_values)
        max_d = np.max(d_values)
        pillar_scores[pillar_name] = {"avg_d": avg_d, "max_d": max_d, "n_feats": len(feats)}
        status = "STRONG" if max_d > 0.8 else "MODERATE" if max_d > 0.5 else "WEAK"
        print(f"  {pillar_name:<18} avg_d={avg_d:.3f}  max_d={max_d:.3f}  [{status}]")
    
    # Pillar bar chart
    if pillar_scores:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(pillar_scores.keys())
        avg_vals = [pillar_scores[n]["avg_d"] for n in names]
        max_vals = [pillar_scores[n]["max_d"] for n in names]
        
        x = np.arange(len(names))
        ax.bar(x - 0.15, avg_vals, 0.3, label="Avg Cohen's d", color="#378ADD", alpha=0.8)
        ax.bar(x + 0.15, max_vals, 0.3, label="Max Cohen's d", color="#E24B4A", alpha=0.8)
        ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.4, label="Large effect")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Cohen's d")
        ax.set_title("Per-Pillar Discriminative Strength")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pillar_analysis.png"), dpi=200)
        plt.close()
    
    return {
        "mahalanobis_distance": float(maha_dist),
        "per_feature_effect_size": {k: float(v) for k, v in per_feature_maha.items()},
        "pillar_scores": {k: {kk: float(vv) for kk, vv in v.items()}
                          for k, v in pillar_scores.items()},
    }


# ============================================================================
# 4. t-SNE VISUALIZATION
# ============================================================================

def compute_tsne(X_train, y_train, X_test, y_test, output_dir,
                 n_samples=2000, perplexity=30):
    """t-SNE visualization of feature space, colored by class and domain."""
    print("\n" + "=" * 60)
    print("4. t-SNE VISUALIZATION")
    print("=" * 60)
    
    # Subsample for speed
    n_train = min(n_samples, len(X_train))
    n_test = min(n_samples, len(X_test))
    
    idx_train = np.random.choice(len(X_train), n_train, replace=False)
    idx_test = np.random.choice(len(X_test), n_test, replace=False)
    
    X_sub = np.vstack([X_train[idx_train], X_test[idx_test]])
    y_sub = np.concatenate([y_train[idx_train], y_test[idx_test]])
    domain = np.concatenate([np.zeros(n_train), np.ones(n_test)])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    
    # PCA reduction first for stability
    pca = PCA(n_components=min(20, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, learning_rate="auto", init="pca")
    X_2d = tsne.fit_transform(X_pca)
    
    # Plot 1: Colored by class (Real/Fake)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for label, color, name in [(0, "#378ADD", "Real"), (1, "#E24B4A", "Fake")]:
        mask = y_sub == label
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=color, alpha=0.4, s=10, label=name)
    axes[0].set_title("t-SNE — Real vs Fake", fontsize=13)
    axes[0].legend(markerscale=3)
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    
    # Plot 2: Colored by domain (Train/Test)
    for d, color, name in [(0, "#1D9E75", "Train (FF++)"),
                            (1, "#D85A30", "Test (CelebDF)")]:
        mask = domain == d
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=color, alpha=0.4, s=10, label=name)
    axes[1].set_title("t-SNE — Domain Shift", fontsize=13)
    axes[1].legend(markerscale=3)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_analysis.png"), dpi=200)
    plt.close()
    
    # Plot 3: Test set only (CelebDF-v2)
    fig, ax = plt.subplots(figsize=(7, 6))
    test_mask = domain == 1
    for label, color, name in [(0, "#378ADD", "Real"), (1, "#E24B4A", "Fake")]:
        mask = test_mask & (y_sub == label)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                  c=color, alpha=0.5, s=15, label=name)
    ax.set_title("t-SNE — CelebDF-v2 Only (Unseen Domain)", fontsize=13)
    ax.legend(markerscale=3)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_celebdf_only.png"), dpi=200)
    plt.close()
    
    print("  t-SNE plots saved.")


# ============================================================================
# 5. FEATURE CORRELATION MATRIX
# ============================================================================

def compute_correlation_matrix(X, feature_names, output_dir):
    """Correlation matrix of all features."""
    print("\n" + "=" * 60)
    print("5. FEATURE CORRELATION MATRIX")
    print("=" * 60)
    
    corr = np.corrcoef(X, rowvar=False)
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, xticklabels=feature_names, yticklabels=feature_names,
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=False, ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=200)
    plt.close()
    
    # Find highly correlated pairs (potential redundancy)
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr[i, j]) > 0.85:
                high_corr_pairs.append(
                    (feature_names[i], feature_names[j], corr[i, j])
                )
    
    if high_corr_pairs:
        print(f"  Highly correlated pairs (|r| > 0.85):")
        for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            print(f"    {f1} <-> {f2}: r={r:.3f}")
    else:
        print("  No highly correlated pairs found (good — low redundancy)")


# ============================================================================
# 6. COMPREHENSIVE SUMMARY REPORT
# ============================================================================

def generate_summary_report(all_results, output_dir):
    """Generate a summary markdown report."""
    report = []
    report.append("# PHANTOM LENS V2 — Feature Space Evaluation Report\n")
    report.append(f"**Features:** {all_results.get('n_features', 'N/A')}\n")
    
    report.append("## 1. Silhouette Score\n")
    sil = all_results.get("silhouette", {})
    report.append(f"- Overall: **{sil.get('silhouette_score', 'N/A'):.4f}**")
    report.append(f"- Real class: {sil.get('silhouette_real', 'N/A'):.4f}")
    report.append(f"- Fake class: {sil.get('silhouette_fake', 'N/A'):.4f}")
    
    report.append("\n## 2. Davies-Bouldin Index\n")
    dbi = all_results.get("dbi", {})
    report.append(f"- DBI: **{dbi.get('dbi_score', 'N/A'):.4f}** (lower is better)")
    
    report.append("\n## 3. Mahalanobis Distance\n")
    maha = all_results.get("mahalanobis", {})
    report.append(f"- Distance: **{maha.get('mahalanobis_distance', 'N/A'):.4f}**")
    
    if "per_feature_effect_size" in maha:
        report.append("\n### Top 10 features by effect size:\n")
        sorted_feats = sorted(maha["per_feature_effect_size"].items(),
                              key=lambda x: x[1], reverse=True)[:10]
        report.append("| Rank | Feature | Cohen's d |")
        report.append("|------|---------|-----------|")
        for rank, (f, d) in enumerate(sorted_feats, 1):
            report.append(f"| {rank} | {f} | {d:.4f} |")
    
    if "pillar_scores" in maha:
        report.append("\n### Per-Pillar Summary:\n")
        report.append("| Pillar | Avg d | Max d | Status |")
        report.append("|--------|-------|-------|--------|")
        for pillar, scores in maha["pillar_scores"].items():
            status = "STRONG" if scores["max_d"] > 0.8 else \
                     "MODERATE" if scores["max_d"] > 0.5 else "WEAK"
            report.append(f"| {pillar} | {scores['avg_d']:.3f} | "
                          f"{scores['max_d']:.3f} | {status} |")
    
    report_text = "\n".join(report)
    
    with open(os.path.join(output_dir, "evaluation_report.md"), "w") as f:
        f.write(report_text)
    
    print(f"\n  Summary report saved to: {output_dir}/evaluation_report.md")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM V3 Feature Space Evaluation — Phantom Lens V2"
    )
    parser.add_argument("--train_csv", type=str, nargs="+",
                        help="Training feature CSVs (optional, for t-SNE domain comparison)")
    parser.add_argument("--test_csv", type=str, nargs="+", required=True,
                        help="Test feature CSVs (primary evaluation target)")
    parser.add_argument("--output_dir", type=str, default="evaluation_v3",
                        help="Output directory")
    parser.add_argument("--tsne_samples", type=int, default=2000,
                        help="Number of samples for t-SNE")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading test data...")
    X_test, y_test, feature_names, test_df = load_and_prepare(args.test_csv)
    print(f"  Test: {len(X_test)} samples, {X_test.shape[1]} features")
    print(f"  Real: {(y_test==0).sum()}, Fake: {(y_test==1).sum()}")
    
    all_results = {"n_features": len(feature_names)}
    
    # 1. Silhouette Score
    sil_results = compute_silhouette(X_test, y_test, args.output_dir)
    all_results["silhouette"] = sil_results
    
    # 2. Davies-Bouldin Index
    dbi_results = compute_dbi(X_test, y_test, args.output_dir)
    all_results["dbi"] = dbi_results
    
    # 3. Mahalanobis Distance
    maha_results = compute_mahalanobis(X_test, y_test, feature_names, args.output_dir)
    all_results["mahalanobis"] = maha_results
    
    # 4. t-SNE (if training data also provided)
    if args.train_csv:
        print("\nLoading training data for t-SNE...")
        X_train, y_train, _, _ = load_and_prepare(args.train_csv)
        print(f"  Train: {len(X_train)} samples")
        compute_tsne(X_train, y_train, X_test, y_test,
                     args.output_dir, n_samples=args.tsne_samples)
    else:
        # t-SNE on test set only
        compute_tsne(X_test, y_test, X_test, y_test,
                     args.output_dir, n_samples=args.tsne_samples)
    
    # 5. Correlation matrix
    compute_correlation_matrix(X_test, feature_names, args.output_dir)
    
    # 6. Summary report
    generate_summary_report(all_results, args.output_dir)
    
    # Save all results as JSON
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Silhouette Score:    {sil_results['silhouette_score']:.4f}")
    print(f"  Davies-Bouldin:      {dbi_results['dbi_score']:.4f}")
    print(f"  Mahalanobis Dist:    {maha_results['mahalanobis_distance']:.4f}")
    print(f"  All outputs in:      {args.output_dir}/")


if __name__ == "__main__":
    main()