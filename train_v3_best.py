#!/usr/bin/env python3
"""
PHANTOM LENS V2 — Training Pipeline
=====================================
Trains classifiers on PRISM V3 physics features.
Supports intra-dataset cross-validation and cross-dataset evaluation.

Classifiers:
  - Logistic Regression (primary, interpretable)
  - Random Forest (ensemble baseline)
  - LightGBM (if available, stronger generalization)

Usage:
    # Train on FF++ and evaluate on CelebDF-v2
    python train_v3.py \
        --train_csv features_ffpp_real.csv features_ffpp_fake.csv \
        --test_csv features_celebdf_real.csv features_celebdf_fake.csv \
        --output_dir results_v3/ \
        --n_folds 5

    # Train on single merged CSV
    python train_v3.py \
        --train_csv train_features.csv \
        --test_csv test_features.csv \
        --output_dir results_v3/
"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, classification_report,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[INFO] LightGBM not available. Using LogisticRegression + RandomForest only.")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_features(csv_paths):
    """Load and merge feature CSVs. Returns DataFrame."""
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    
    # Drop rows with too many NaNs
    feature_cols = [c for c in merged.columns if c.startswith("s_") or c.startswith("t_")]
    nan_frac = merged[feature_cols].isna().mean(axis=1)
    merged = merged[nan_frac < 0.5].reset_index(drop=True)
    
    # Fill remaining NaNs with column median
    for col in feature_cols:
        merged[col] = merged[col].fillna(merged[col].median())
    
    # Replace inf
    merged = merged.replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        merged[col] = merged[col].fillna(merged[col].median())
    
    return merged


def prepare_xy(df):
    """Extract X (features) and y (labels) from DataFrame."""
    feature_cols = sorted([c for c in df.columns
                           if c.startswith("s_") or c.startswith("t_")])
    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values.astype(int)
    return X, y, feature_cols


# ============================================================================
# TRAINING WITH CROSS-VALIDATION
# ============================================================================

def train_and_evaluate(X_train, y_train, X_test, y_test,
                       feature_names, output_dir, n_folds=5):
    """Train classifiers with cross-validation, evaluate on test set."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    # ================================================================
    # Define classifiers
    # ================================================================
    classifiers = {
        "LogisticRegression": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    
    if HAS_LGBM:
        classifiers["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=20,
            class_weight="balanced", random_state=42,
            verbose=-1
        )
    
    results = {}
    
    # ================================================================
    # Cross-validation on training set
    # ================================================================
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION ON TRAINING SET")
    print("=" * 60)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for clf_name, clf_template in classifiers.items():
        print(f"\n--- {clf_name} ---")
        fold_aucs = []
        fold_accs = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
            Xtr, Xval = X_train_scaled[train_idx], X_train_scaled[val_idx]
            ytr, yval = y_train[train_idx], y_train[val_idx]
            
            from sklearn.base import clone
            clf = clone(clf_template)
            clf.fit(Xtr, ytr)
            
            y_prob = clf.predict_proba(Xval)[:, 1]
            fold_auc = roc_auc_score(yval, y_prob)
            fold_acc = accuracy_score(yval, clf.predict(Xval))
            fold_aucs.append(fold_auc)
            fold_accs.append(fold_acc)
        
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        mean_acc = np.mean(fold_accs)
        
        print(f"  CV AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  CV Acc: {mean_acc:.4f}")
        
        results[clf_name] = {
            "cv_auc_mean": mean_auc,
            "cv_auc_std": std_auc,
            "cv_acc_mean": mean_acc,
            "cv_fold_aucs": fold_aucs,
        }
    
    # ================================================================
    # Train on full training set, evaluate on test set
    # ================================================================
    print("\n" + "=" * 60)
    print("CROSS-DATASET EVALUATION (TEST SET)")
    print("=" * 60)
    
    best_auc = 0
    best_clf_name = None
    
    for clf_name, clf_template in classifiers.items():
        print(f"\n--- {clf_name} ---")
        
        from sklearn.base import clone
        clf = clone(clf_template)
        clf.fit(X_train_scaled, y_train)
        
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        y_pred = clf.predict(X_test_scaled)
        
        test_auc = roc_auc_score(y_test, y_prob)
        test_acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )
        
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")
        print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(classification_report(y_test, y_pred,
                                     target_names=["Real", "Fake"]))
        
        results[clf_name].update({
            "test_auc": test_auc,
            "test_acc": test_acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
        })
        
        # Save model
        model_path = os.path.join(output_dir, f"model_{clf_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"{clf_name} (AUC={test_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {clf_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"roc_{clf_name}.png"), dpi=150)
        plt.close()
        
        if test_auc > best_auc:
            best_auc = test_auc
            best_clf_name = clf_name
        
        # Feature importance (for tree-based models)
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            sorted_idx = np.argsort(imp)[::-1][:20]
            
            plt.figure(figsize=(8, 6))
            plt.barh(range(len(sorted_idx)),
                     imp[sorted_idx][::-1],
                     tick_label=[feature_names[i] for i in sorted_idx][::-1])
            plt.xlabel("Importance")
            plt.title(f"Top 20 Features — {clf_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                                      f"feature_importance_{clf_name}.png"),
                        dpi=150)
            plt.close()
        
        # For LogisticRegression, show coefficient magnitudes
        if hasattr(clf, "coef_"):
            coefs = np.abs(clf.coef_[0])
            sorted_idx = np.argsort(coefs)[::-1][:20]
            
            plt.figure(figsize=(8, 6))
            plt.barh(range(len(sorted_idx)),
                     coefs[sorted_idx][::-1],
                     tick_label=[feature_names[i] for i in sorted_idx][::-1])
            plt.xlabel("|Coefficient|")
            plt.title(f"Top 20 Features — {clf_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                                      f"feature_coefs_{clf_name}.png"),
                        dpi=150)
            plt.close()
    
    # ================================================================
    # Cohen's d effect size per feature
    # ================================================================
    print("\n" + "=" * 60)
    print("COHEN'S d EFFECT SIZE (TEST SET)")
    print("=" * 60)
    
    cohens_d = {}
    real_mask = y_test == 0
    fake_mask = y_test == 1
    
    for i, fname in enumerate(feature_names):
        real_vals = X_test_scaled[real_mask, i]
        fake_vals = X_test_scaled[fake_mask, i]
        
        n1, n2 = len(real_vals), len(fake_vals)
        if n1 < 2 or n2 < 2:
            cohens_d[fname] = 0.0
            continue
        
        mean_diff = real_vals.mean() - fake_vals.mean()
        pooled_std = np.sqrt(
            ((n1 - 1) * real_vals.std()**2 + (n2 - 1) * fake_vals.std()**2)
            / (n1 + n2 - 2)
        )
        d = abs(mean_diff) / (pooled_std + 1e-8)
        cohens_d[fname] = d
    
    sorted_d = sorted(cohens_d.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Rank':>4} {'Feature':<35} {'Cohen d':>8} {'Effect':>12}")
    print("-" * 65)
    for rank, (fname, d) in enumerate(sorted_d[:25], 1):
        if d > 0.8:
            effect = "Large"
        elif d > 0.5:
            effect = "Medium"
        elif d > 0.2:
            effect = "Small"
        else:
            effect = "Negligible"
        print(f"{rank:4d} {fname:<35} {d:8.4f} {effect:>12}")
    
    # Plot Cohen's d
    top_n = min(30, len(sorted_d))
    names = [x[0] for x in sorted_d[:top_n]]
    values = [x[1] for x in sorted_d[:top_n]]
    colors = ["#e24b4a" if v > 0.8 else "#ef9f27" if v > 0.5
              else "#378add" for v in values]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), values[::-1], color=colors[::-1])
    plt.yticks(range(top_n), names[::-1], fontsize=9)
    plt.axvline(x=0.8, color="red", linestyle="--", alpha=0.5, label="Large effect")
    plt.axvline(x=0.5, color="orange", linestyle="--", alpha=0.5, label="Medium effect")
    plt.xlabel("Cohen's d")
    plt.title("Feature Discriminative Strength (Cohen's d)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cohens_d.png"), dpi=150)
    plt.close()
    
    # ================================================================
    # Save results
    # ================================================================
    results["best_classifier"] = best_clf_name
    results["best_test_auc"] = best_auc
    results["cohens_d"] = cohens_d
    results["n_features"] = len(feature_names)
    results["feature_names"] = feature_names
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_clf_name} — Test AUC: {best_auc:.4f}")
    print(f"All results saved to: {output_dir}")
    print(f"{'=' * 60}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM V3 Training Pipeline — Phantom Lens V2"
    )
    parser.add_argument("--train_csv", type=str, nargs="+", required=True,
                        help="Training feature CSV files")
    parser.add_argument("--test_csv", type=str, nargs="+", required=True,
                        help="Test feature CSV files (cross-dataset)")
    parser.add_argument("--output_dir", type=str, default="results_v3",
                        help="Output directory for results")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of cross-validation folds")
    args = parser.parse_args()
    
    print("Loading training data...")
    train_df = load_features(args.train_csv)
    print(f"  Training samples: {len(train_df)} "
          f"(Real: {(train_df['label']==0).sum()}, "
          f"Fake: {(train_df['label']==1).sum()})")
    
    print("Loading test data...")
    test_df = load_features(args.test_csv)
    print(f"  Test samples: {len(test_df)} "
          f"(Real: {(test_df['label']==0).sum()}, "
          f"Fake: {(test_df['label']==1).sum()})")
    
    X_train, y_train, feature_names = prepare_xy(train_df)
    X_test, y_test, _ = prepare_xy(test_df)
    
    print(f"\nFeature dimensions: {X_train.shape[1]}")
    print(f"Features: {feature_names}")
    
    results = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        feature_names, args.output_dir, args.n_folds
    )


if __name__ == "__main__":
    main()