#!/usr/bin/env python3
"""Leak-free missing-value imputation (fixes guide catch #6 / audit M1).

The retired idiom computed fillna(median) over an ENTIRE feature CSV (train+val+test)
BEFORE partitioning, so test rows influenced the medians used on train — a (mild)
imputation leak. These helpers compute medians from the TRAIN partition ONLY and apply
them unchanged to val/test (and to external zero-shot sets), matching §0.

NOTE: the extracted 50-D matrices contain no missing cells and the residual CSVs contain
~1 cell/file, so this fix is a code-correctness/defensibility change; it reproduces every
locked number (verified to >=4 decimals). It is NOT a re-baseline.
"""
import numpy as np, pandas as pd
from protocol import make_splits


def _numeric_nan(df, cols):
    d = df.copy()
    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return d


def split_impute(df, feature_cols, id2split=None, path_col="video_path"):
    """Partition `df` by identity, then fill missing feature cells with TRAIN-partition
    medians ONLY. Returns (imputed_df_with_partition, train_median_series)."""
    d = make_splits(_numeric_nan(df, feature_cols), id2split=id2split, path_col=path_col)
    med = d.loc[d.partition == "train", feature_cols].median()
    d[feature_cols] = d[feature_cols].fillna(med)
    return d, med


def impute_with(df, feature_cols, train_median):
    """Impute an external / zero-shot set (e.g. Celeb-DF, WildDeepfake) using a supplied
    TRAIN median (from the FF++ training set the classifier was fit on) — never its own rows."""
    d = _numeric_nan(df, feature_cols)
    d[feature_cols] = d[feature_cols].fillna(train_median)
    return d


def pooled_train_median(imputed_frames, feature_cols):
    """Median over the pooled TRAIN partitions of already-split frames (for zero-shot imputation)."""
    tr = pd.concat([f[f.partition == "train"] for f in imputed_frames], ignore_index=True)
    return tr[feature_cols].replace([np.inf, -np.inf], np.nan).median()
