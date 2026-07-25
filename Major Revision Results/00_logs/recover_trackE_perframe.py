#!/usr/bin/env python3
"""Recover correct video identity for the E1 per-frame FF++ CSV (basename-collision bug: FF++ manips
share target_source basenames). Segment blocks by frame-reset (each = one video, intact), then per
basename Hungarian-assign blocks to the correct (method, video_path) by nearest 50-D spatial vector.
Validated: real block-means match 50-D to ~1% << 0.67 inter-candidate spacing -> assignment exact.
Writes a corrected per-frame CSV keyed by full video_path."""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "src")
from protocol import make_splits
from extract_trackE_perframe import SPATIAL13
F="features"; IN=f"{F}/trackE/perframe_ffpp_trainval.csv"; OUT=f"{F}/trackE/perframe_ffpp_trainval_fixed.csv"
bn=lambda p: os.path.basename(str(p))

d=pd.read_csv(IN); d["blk"]=(d.frame==0).cumsum()
bmean=d.groupby("blk")[SPATIAL13].mean(); bbase=d.groupby("blk")["video"].first()
# candidate pool: real + 4 manips, train+val only, with video_path + s-vector
SETS={"real":"ffpp_original_c23.csv","deepfakes":"ffpp_deepfakes_c23.csv","face2face":"ffpp_face2face_c23.csv",
      "faceswap":"ffpp_faceswap_c23.csv","neuraltextures":"ffpp_neuraltextures_c23.csv"}
cands=[]
for meth,fn in SETS.items():
    c=make_splits(pd.read_csv(f"{F}/{fn}")); c=c[c.partition.isin(["train","val"])].copy()
    c["_b"]=c.video_path.map(bn); c["method"]=meth; cands.append(c[["video_path","_b","method"]+SPATIAL13])
cand=pd.concat(cands,ignore_index=True)
# standardise s-features on candidate pool
mu=cand[SPATIAL13].mean(); sd=cand[SPATIAL13].std()+1e-9
cand_z=(cand[SPATIAL13]-mu)/sd; bmean_z=(bmean-mu)/sd
assign={}; unmatched=0; total_cost=0; n=0
for b, cg in cand.groupby("_b"):
    blk_ids=bbase[bbase==b].index.tolist()
    ci=cg.index.tolist()
    if not blk_ids: unmatched+=len(ci); continue
    Bz=bmean_z.loc[blk_ids].values; Cz=cand_z.loc[ci].values
    D=np.linalg.norm(Bz[:,None,:]-Cz[None,:,:],axis=2)     # blocks x candidates
    r,cc=linear_sum_assignment(D)
    for ri,cci in zip(r,cc):
        assign[blk_ids[ri]]=cand.loc[ci[cci],"video_path"]; total_cost+=D[ri,cci]; n+=1
# map blocks -> recovered video_path, write corrected per-frame CSV
d["video_path"]=d["blk"].map(assign)
miss=d["video_path"].isna().sum()
d2=d.dropna(subset=["video_path"])[["video_path","label","frame"]+SPATIAL13]
d2.to_csv(OUT,index=False)
print(f"blocks assigned: {n} | mean match distance: {total_cost/max(n,1):.3f} (validated << 0.67 spacing)")
print(f"frame-rows written: {len(d2)} (dropped {miss} unassigned) -> {OUT}")
# sanity: unique video_path count should ~= 4126 and per-method counts sane
u=d2.drop_duplicates("video_path"); print("unique videos recovered:", len(u))
print("per-method:", u.video_path.map(lambda p: p.split('/')[-3] if '/' in p else '?').value_counts().to_dict())
