#!/usr/bin/env python3
"""Run the Track-C ROI extractor (extract_roi_features.process_video) over an explicit
video manifest (video_path,label) instead of a directory — used to give Celeb-DF the ROI/G1
features so the 53-D model has cross-dataset coverage (Track D). Deterministic; committed."""
import argparse, csv, sys, os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))
import extract_roi_features as R
import roi_config as RC
def _w(a): return R.process_video(*a)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--max_frames", type=int, default=150); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    man = pd.read_csv(a.manifest); tasks = [(r.video_path, int(r.label), a.max_frames) for r in man.itertuples()]
    print(f"ROI/G1 over manifest: {len(tasks)} videos -> {a.output}", flush=True)
    hdr = ["video_path", "label"] + RC.ROI_FEATURE_NAMES
    out = open(a.output, "w", newline=""); w = csv.DictWriter(out, fieldnames=hdr); w.writeheader(); ok=fail=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_w, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            if r: w.writerow(r); out.flush(); ok+=1
            else: fail+=1
            if (ok+fail)%200==0: print(f"  {ok+fail}/{len(tasks)}",flush=True)
    out.close(); print(f"Done. ok={ok} fail={fail} -> {a.output}",flush=True)
