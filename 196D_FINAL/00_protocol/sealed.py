#!/usr/bin/env python3
"""Track D — SEALED-SET GATING (anti-overfitting protocol).

The FF++ test identities and the Celeb-DF `test` half (splits/celebdf_dev_test.json) are SEALED:
evaluated exactly ONCE, at the very end, after the feature set is frozen. This module makes the
sealed sets impossible to touch accidentally — any access raises unless the caller explicitly
unseals, and every unseal is appended to a log so the number of sealed evaluations is auditable.

Dev iteration must use ONLY: FF++ train/val, and Celeb-DF `dev`.  (WildDeepfake = secondary dev signal.)
"""
import os, re, json, datetime

_SPLIT = os.path.join(os.path.dirname(__file__), "..", "splits", "celebdf_dev_test.json")
_LOG = os.path.join(os.path.dirname(__file__), "..", "Major Revision Results", "07_summary", "sealed_eval_log.txt")

def _load():
    return json.load(open(_SPLIT))

def celebdf_partition(df, path_col="video_path"):
    """Add a 'ct_partition' column (dev / test / DROP) to a Celeb-DF feature DataFrame, per the
    sealed split. Use df[df.ct_partition=='dev'] for iteration; 'test' is sealed (see unseal)."""
    s = _load()
    def part(p):
        b = os.path.basename(str(p)); ids = re.findall(r"id(\d+)", b)
        if ids:
            ps = {s["id2split"][i] for i in ids if i in s["id2split"]}
            return ps.pop() if len(ps) == 1 else "DROP"
        return s["youtube2split"].get(b, "DROP")
    df = df.copy(); df["ct_partition"] = df[path_col].map(part)
    return df

def unseal(name, allow_sealed=False):
    """Gate a SEALED evaluation set. Raises unless explicitly unsealed (allow_sealed=True OR env
    TRACKD_ALLOW_SEALED=1). Every successful unseal is logged — this is what the protocol's
    'sealed evaluations performed: 1' count is read from. Call ONLY in the single Phase-4 eval."""
    if not (allow_sealed or os.environ.get("TRACKD_ALLOW_SEALED") == "1"):
        raise RuntimeError(
            f"SEALED SET '{name}' is LOCKED. Dev iteration must not touch it. To run the single "
            f"final Phase-4 evaluation, pass allow_sealed=True (or set TRACKD_ALLOW_SEALED=1).")
    os.makedirs(os.path.dirname(_LOG), exist_ok=True)
    with open(_LOG, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()}\tUNSEAL\t{name}\n")
    return True

def sealed_eval_count(name=None):
    """Number of times a sealed set has been unsealed (for the LOCKED_NUMBERS.md audit line)."""
    if not os.path.exists(_LOG):
        return 0
    return sum(1 for l in open(_LOG)
               if "\tUNSEAL\t" in l and (name is None or l.rstrip().endswith("\t" + name)))

if __name__ == "__main__":
    import pandas as pd
    df = celebdf_partition(pd.read_csv("features/celebdf_features.csv"))
    print("ct_partition counts:", df.ct_partition.value_counts().to_dict())
    try:
        unseal("celebdf_test"); print("ERROR: unseal should have raised")
    except RuntimeError as e:
        print("default access BLOCKED ✓ —", str(e)[:60], "...")
    print("sealed_eval_count:", sealed_eval_count())
