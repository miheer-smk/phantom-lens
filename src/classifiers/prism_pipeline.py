"""PRISM-50 pipeline: the frozen protocol, path-portable.

Everything a third party needs to refit and rescore from the released feature matrices.
No absolute paths: the feature directory is passed in or read from $PRISM_FEATURES.
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np, pandas as pd

MANIPULATIONS = ["deepfakes", "face2face", "faceswap", "neuraltextures"]


def feature_dir(explicit: str | None = None) -> Path:
    d = explicit or os.environ.get("PRISM_FEATURES")
    if not d:
        raise SystemExit(
            "Set PRISM_FEATURES to the directory holding the released feature CSVs, "
            "or pass --features. See docs/DATASET_ACCESS.md.")
    p = Path(d).expanduser()
    if not p.is_dir():
        raise SystemExit(f"PRISM_FEATURES is not a directory: {p}")
    return p


def load_config(repo_root: Path) -> dict:
    import yaml
    return yaml.safe_load(open(repo_root / "configs" / "lightgbm.yaml"))["params"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """ALPHABETICAL, not CSV order. Getting this wrong moves every tree split."""
    return sorted(c for c in df.columns if c[:2] in ("s_", "t_"))


def load_split(repo_root: Path) -> dict:
    csv = repo_root / "splits" / "ffpp_identity_split.csv"
    d = pd.read_csv(csv, dtype={"source_video_id": str})
    return dict(zip(d.source_video_id, d.partition))


def assign_partition(df: pd.DataFrame, id2split: dict) -> pd.DataFrame:
    """A pristine clip by its own id; a manipulated clip <target>_<source> by its target."""
    def sid(p):
        stem = os.path.splitext(os.path.basename(str(p)))[0]
        return stem.split("_")[0]
    out = df.copy()
    out["source_video_id"] = out.video_path.map(sid)
    out["partition"] = out.source_video_id.map(id2split)
    return out


def assert_no_identity_overlap(frames: list[tuple[pd.DataFrame, str]]) -> None:
    seen: dict[str, str] = {}
    for d, part in frames:
        for i in d.source_video_id:
            if i in seen and seen[i] != part:
                raise AssertionError(f"identity {i} appears in both {seen[i]} and {part}")
            seen[i] = part


def train_median_impute(train: pd.DataFrame, others: list[pd.DataFrame], cols: list[str]):
    """Medians from the TRAIN partition only, applied unchanged elsewhere. Never fit on test."""
    tr = train.copy()
    for c in cols:
        tr[c] = pd.to_numeric(tr[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = tr[cols].median()
    tr[cols] = tr[cols].fillna(med)
    outs = []
    for d in others:
        o = d.copy()
        for c in cols:
            o[c] = pd.to_numeric(o[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        o[cols] = o[cols].fillna(med)
        outs.append(o)
    return tr, outs, med


def fit_prism(Xtr, ytr, params: dict):
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    sc = StandardScaler().fit(Xtr)
    clf = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != "verbose"}, verbose=-1)
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def score(sc, clf, X):
    return clf.predict_proba(sc.transform(X))[:, 1]


# ---------------------------------------------------------------------------
# Shared evaluation helpers used by the experiments/ runners
# ---------------------------------------------------------------------------

def grouped_ci(y, scores, groups, n_boot: int = 2000, seed: int = 42):
    """Identity-grouped cluster bootstrap (R1-C7).

    Duplicated groups REPLICATE their rows; they are never deduplicated by a boolean mask.
    Resamples missing a class are skipped and counted.
    Returns (lo, hi, n_skipped, n_groups).
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(scores); g = np.asarray(groups)
    uniq = np.unique(g)
    g2r = {x: np.flatnonzero(g == x) for x in uniq}
    rng = np.random.default_rng(seed)
    out, skipped = [], 0
    for _ in range(n_boot):
        idx = np.concatenate([g2r[x] for x in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            skipped += 1; continue
        out.append(roc_auc_score(y[idx], s[idx]))
    if not out:
        return None, None, skipped, len(uniq)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), skipped, len(uniq)


def metrics(y, p, groups=None, seed: int = 42, with_ci: bool = True):
    """AUC (+ grouped CI), macro-F1, MCC, class recalls, n."""
    from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, recall_score
    y = np.asarray(y); p = np.asarray(p); pred = (p >= 0.5).astype(int)
    m = dict(auc=round(float(roc_auc_score(y, p)), 4),
             macro_f1=round(float(f1_score(y, pred, average="macro")), 4),
             mcc=round(float(matthews_corrcoef(y, pred)), 4),
             real_recall=round(float(recall_score(y, pred, pos_label=0)), 4),
             fake_recall=round(float(recall_score(y, pred, pos_label=1)), 4),
             n=int(len(y)), n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()))
    if with_ci and groups is not None:
        lo, hi, sk, ng = grouped_ci(y, p, groups, seed=seed)
        m["grouped_ci"] = [round(lo, 4), round(hi, 4)] if lo is not None else None
        m["ci_skipped_resamples"] = sk; m["n_groups"] = ng
    return m


def descriptor_groups(repo_root: Path) -> dict:
    """The 20 implementation-level groups (Table A2)."""
    import yaml
    return yaml.safe_load(open(repo_root / "configs" / "prism50.yaml"))["descriptor_groups"]


def load_ffpp(F: Path, id2split: dict, compression: str = "c23"):
    """Load the FF++ matrices for one compression level, partitioned by identity."""
    raw = {"real": pd.read_csv(F / f"ffpp_original_{compression}.csv")}
    for m in MANIPULATIONS:
        raw[m] = pd.read_csv(F / f"ffpp_{m}_{compression}.csv")
    return {k: assign_partition(v, id2split) for k, v in raw.items()}
