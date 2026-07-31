#!/usr/bin/env python3
"""Leakage-free evaluation protocol for PRISM (Track A).

FF++ clip naming:
  real:       <id>.mp4                (single source identity)
  manipulated:<target>_<source>.mp4   (two source identities)
Official FF++ split assigns each of the 1000 source identities to exactly one of
train/val/test, and guarantees target & source of any manipulated clip share a split.

make_splits() maps every extracted-feature row to a partition by IDENTITY, and
assert_no_identity_overlap() fails LOUDLY if any identity crosses splits.
This module MUST be imported and its assertion run at the start of every experiment.
"""
import json, os, re
import pandas as pd

_SPLIT_JSON = os.path.join(os.path.dirname(__file__), "..", "splits", "ffpp_official_split.json")

def load_id2split(path=_SPLIT_JSON):
    d = json.load(open(path))
    return d["id2split"]

def clip_identities(filename):
    """Return the set of source identities a clip depends on."""
    stem = re.sub(r"\.(mp4|avi|mov)$", "", os.path.basename(str(filename)), flags=re.I)
    parts = stem.split("_")
    # real '<id>' -> {id}; manipulated '<target>_<source>' -> {target, source}
    ids = [p for p in parts[:2] if p.isdigit()]
    return set(ids) if ids else {stem}

def assign_partition(filename, id2split):
    """Partition of a clip = the (unique) split of its identities. None if unknown."""
    ids = clip_identities(filename)
    parts = {id2split.get(i) for i in ids}
    parts.discard(None)
    if len(parts) == 1:
        return parts.pop()
    if len(parts) == 0:
        return None            # identity not in split table
    raise ValueError(f"IDENTITY-LEVEL LEAK: {filename} spans splits {parts}")

def make_splits(df, id2split=None, path_col="video_path"):
    """Add a 'partition' column (train/val/test/None) to a feature DataFrame by identity."""
    if id2split is None: id2split = load_id2split()
    df = df.copy()
    df["partition"] = df[path_col].map(lambda p: assign_partition(p, id2split))
    return df

def assert_no_identity_overlap(dfs_by_name, id2split=None, path_col="video_path"):
    """Assert no source identity appears in more than one partition across all frames.
    dfs_by_name: dict name-> (df, partition_label) OR list of (df, partition_label)."""
    if id2split is None: id2split = load_id2split()
    seen = {}  # identity -> partition
    items = dfs_by_name.items() if isinstance(dfs_by_name, dict) else dfs_by_name
    for name, (df, part) in (items if isinstance(dfs_by_name, dict)
                             else [(f"df{i}", v) for i, v in enumerate(dfs_by_name)]):
        for fn in df[path_col]:
            for i in clip_identities(fn):
                if i in seen and seen[i] != part:
                    raise AssertionError(
                        f"IDENTITY OVERLAP: id {i} in both '{seen[i]}' and '{part}' (via {name}/{fn})")
                seen[i] = part
    return True   # no overlap

if __name__ == "__main__":
    # self-test on extracted CSVs
    import glob
    id2split = load_id2split()
    print("split counts:", json.load(open(_SPLIT_JSON))["counts"])
    for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "features", "ffpp_*_c23.csv"))):
        df = make_splits(pd.read_csv(f), id2split)
        vc = df["partition"].value_counts(dropna=False).to_dict()
        print(f"  {os.path.basename(f):28s} {vc}")
