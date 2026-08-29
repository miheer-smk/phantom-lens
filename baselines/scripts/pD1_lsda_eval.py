#!/usr/bin/env python
"""D1 / Phase 7.6 - evaluate the trained LSDA on FF++ c23 test (per manipulation) and zero-shot Celeb-DF.

Deliberately does NOT use DeepfakeBench's test.py: the brief requires that every frame-level deep
baseline aggregate IDENTICALLY, p_V = (1/m) sum_t p_t over the same sampled frames, so that LSDA,
Xception and LAA-Net are directly comparable. This evaluator applies that aggregation itself.
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R=Path.home()/"prism_r2"; DFB=R/"baselines"/"DeepfakeBench"
sys.path.insert(0,str(DFB)); sys.path.insert(0,str(DFB/"training"))

def grouped_ci(y,s,g,n=2000,seed=42):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y);s=np.asarray(s);g=np.asarray(g).astype(str)
    assert len(y)==len(s)==len(g), (
        f"MISALIGNED: y={len(y)} s={len(s)} g={len(g)}. Scores and labels must be indexed "
        f"identically or the AUC is computed against the wrong labels.")
    u=np.unique(g); g2={x:np.flatnonzero(g==x) for x in u}
    rng=np.random.default_rng(seed); out=[];sk=0
    for _ in range(n):
        i=np.concatenate([g2[x] for x in rng.choice(u,len(u),True)])
        if len(np.unique(y[i]))<2: sk+=1; continue
        out.append(roc_auc_score(y[i],s[i]))
    return (float(np.percentile(out,2.5)),float(np.percentile(out,97.5)),sk,len(u)) if out else (None,None,sk,len(u))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights",required=True)
    ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--out",default=str(R/"results"/"PD1_lsda_eval.json"))
    ap.add_argument("--scores",default=str(R/"results"/"lsda_per_video_scores.csv"))
    ap.add_argument("--validate-only",action="store_true")
    a=ap.parse_args()
    import torch, yaml, cv2
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, recall_score
    os.chdir(DFB)
    from detectors import DETECTOR
    cfg=yaml.safe_load(open("training/config/detector/lsda_prism.yaml"))
    cfg.update(yaml.safe_load(open("training/config/train_config.yaml")))
    cfg["cuda"]=torch.cuda.is_available()
    DEV="cuda" if cfg["cuda"] else "cpu"
    model=DETECTOR[cfg["model_name"]](cfg)
    sd=torch.load(a.weights,map_location="cpu")
    if isinstance(sd,dict) and "state_dict" in sd: sd=sd["state_dict"]
    sd={k.replace("module.",""):v for k,v in sd.items()}
    miss,unexp=model.load_state_dict(sd,strict=False)
    print(f"LSDA checkpoint: {len(miss)} missing, {len(unexp)} unexpected keys")
    model.eval().to(DEV)
    RES=cfg.get("resolution",256)
    # The same class of bug as DEV-010: cfg.get(...) with a default silently substitutes
    # [0.5,0.5,0.5] if the training config lacks the key, feeding the model inputs it was never
    # trained on. Require the keys to be present rather than defaulting.
    for _k in ("mean", "std"):
        if _k not in cfg:
            raise KeyError(f"training config {cfg.get('model_name','?')} has no '{_k}'. Refusing "
                           f"to assume a default - a wrong normalisation returns plausible scores "
                           f"silently. See DEV-010.")
    MEAN=np.array(cfg["mean"],np.float32); STD=np.array(cfg["std"],np.float32)
    assert MEAN.shape==STD.shape==(3,) and not np.allclose(STD,0), "bad normalisation in config"
    print(f"normalisation from training config: MEAN={list(np.round(MEAN,4))} STD={list(np.round(STD,4))}")
    class Frames(Dataset):
        def __init__(self,paths): self.p=paths
        def __len__(self): return len(self.p)
        def __getitem__(self,i):
            im=cv2.imread(self.p[i])
            if im is None: im=np.zeros((RES,RES,3),np.uint8)
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            if im.shape[:2]!=(RES,RES): im=cv2.resize(im,(RES,RES),interpolation=cv2.INTER_CUBIC)
            x=(im.astype(np.float32)/255.0-MEAN)/STD
            return torch.from_numpy(x.transpose(2,0,1)), i
    def score_frames(paths):
        dl=DataLoader(Frames(paths),batch_size=a.batch,num_workers=4,pin_memory=True)
        out=np.zeros(len(paths),np.float32)
        with torch.no_grad():
            for xb,idx in dl:
                # LSDA's forward signature requires 'label' even at inference. We pass a CONSTANT
                # dummy (all zeros) and separately assert the output is invariant to it, so no label
                # information can reach the score. See the invariance check below.
                xb=xb.to(DEV,non_blocking=True)
                r=model({"image":xb,"label":torch.zeros(len(xb),dtype=torch.long,device=DEV)},inference=True)
                p=r["prob"] if isinstance(r,dict) else r
                out[idx.numpy()]=p.detach().float().cpu().numpy().ravel()
        return out
    JD=DFB/"preprocessing"/"dataset_json_v6"
    ff=json.load(open(JD/"FaceForensics++.json"))["FaceForensics++"]
    if a.validate_only:
        sample=list(ff["FF-real"]["test"]["c23"].values())[0]["frames"][:a.batch]
        p=score_frames(sample)
        print(f"forward pass OK: {len(p)} frames, p range [{p.min():.4f},{p.max():.4f}]")
        # LEAKAGE CHECK: the score must not depend on the dummy label we are forced to pass.
        import cv2 as _cv
        def _batch(paths):
            xs=[]
            for q in paths:
                im=_cv.imread(q); im=_cv.cvtColor(im,_cv.COLOR_BGR2RGB)
                if im.shape[:2]!=(RES,RES): im=_cv.resize(im,(RES,RES),interpolation=_cv.INTER_CUBIC)
                xs.append(((im.astype(np.float32)/255.0-MEAN)/STD).transpose(2,0,1))
            return torch.from_numpy(np.stack(xs)).to(DEV)
        xb=_batch(sample)
        with torch.no_grad():
            p0=model({"image":xb,"label":torch.zeros(len(xb),dtype=torch.long,device=DEV)},
                     inference=True)["prob"].float().cpu().numpy().ravel()
            p1=model({"image":xb,"label":torch.ones(len(xb),dtype=torch.long,device=DEV)},
                     inference=True)["prob"].float().cpu().numpy().ravel()
        d=float(np.abs(p0-p1).max())
        print(f"LABEL-INVARIANCE CHECK: max|p(label=0) - p(label=1)| = {d:.3e}  "
              f"{'PASS - no label leakage' if d==0 else '*** FAIL - the score depends on the label ***'}")
        return 0 if d==0 else 1
    KEY={"deepfakes":"FF-DF","face2face":"FF-F2F","faceswap":"FF-FS","neuraltextures":"FF-NT"}
    res={"generated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
         "model":"LSDA (CVPR 2024), DeepfakeBench","weights":a.weights,
         "aggregation":"p_V = (1/m) sum_t p_t - IDENTICAL to Xception and LAA-Net","per_manipulation":{}}
    rows=[]
    def collect(block,label,vids=None):
        out=[]
        for vid,rec in block.items():
            if vids is not None and vid not in vids: continue
            out.append((vid,label,rec["frames"]))
        return out
    real_te=collect(ff["FF-real"]["test"]["c23"],0)
    print(f"{'target':16s} {'AUC':>7s} {'grouped CI':>18s} {'mF1':>7s} {'MCC':>7s}   n")
    for man,k in KEY.items():
        fake_te=collect(ff[k]["test"]["c23"],1)
        items=real_te+fake_te
        paths=[p for _,_,fr in items for p in fr]; owner=[i for i,(_,_,fr) in enumerate(items) for _ in fr]
        pf=score_frames(paths)
        agg=pd.DataFrame({"o":owner,"p":pf}).groupby("o")["p"].mean()
        # index y and g by the owners that actually produced frames. An item whose frame list
        # is empty contributes no row to agg, and taking y over ALL items would silently pair
        # every later score with the wrong label.
        keep=agg.index.to_numpy(); s=agg.values
        y=np.array([items[i][1] for i in keep])
        g=np.array([items[i][0].split("_")[0] for i in keep])
        if len(keep)!=len(items):
            print(f"  {man}: {len(items)-len(keep)} item(s) had no frames and are excluded")
        pred=(s>=0.5).astype(int); lo,hi,sk,ng=grouped_ci(y,s,g)
        m=dict(auc=round(float(roc_auc_score(y,s)),4),
               grouped_ci=[round(lo,4),round(hi,4)] if lo is not None else None,
               macro_f1=round(float(f1_score(y,pred,average="macro")),4),
               mcc=round(float(matthews_corrcoef(y,pred)),4),
               n=int(len(y)),n_real=int((y==0).sum()),n_fake=int((y==1).sum()),n_groups=ng)
        res["per_manipulation"][man]=m
        json.dump(res,open(a.out,"w"),indent=1)   # partial save: a later crash must not lose this
        for i,sc in zip(keep,s): rows.append(dict(video=items[i][0],label=items[i][1],p=float(sc),target=man,dataset="FFpp"))
        print(f"{man:16s} {m['auc']:7.4f} {str(m['grouped_ci']):>18s} {m['macro_f1']:7.4f} {m['mcc']:7.4f}  {m['n']}")
    cdj=JD/"Celeb-DF-v2.json"
    if cdj.exists():
        cd=json.load(open(cdj))["Celeb-DF-v2"]
        items=[]
        for sub,lab in (("CelebDFv2_real",0),("CelebDFv2_fake",1)):
            for mode in ("test","train","val"):
                blk=cd.get(sub,{}).get(mode,{})
                blk=blk.get("c23",blk) if isinstance(blk,dict) else {}
                for vid,rec in (blk.items() if isinstance(blk,dict) else []):
                    if isinstance(rec,dict) and "frames" in rec: items.append((vid,lab,rec["frames"]))
        # DeepfakeBench's Celeb-DF json lists the SAME videos under train, val and test - the
        # frame lists are byte-identical. Iterating all three counts most videos three times and,
        # because the real class is split 588/890/890, reweights it unevenly rather than
        # cancelling out. Deduplicate by video id; the full release is 890 real + 5639 fake.
        _seen=set(); _dedup=[]
        for it in items:
            if it[0] in _seen: continue
            _seen.add(it[0]); _dedup.append(it)
        _ndup=len(items)-len(_dedup)
        if _ndup: print(f"  celebdf: {_ndup} duplicate listings removed -> {len(_dedup)} unique videos")
        items=_dedup
        if items:
            paths=[p for _,_,fr in items for p in fr]; owner=[i for i,(_,_,fr) in enumerate(items) for _ in fr]
            pf=score_frames(paths)
            agg=pd.DataFrame({"o":owner,"p":pf}).groupby("o")["p"].mean()
            keep=agg.index.to_numpy(); s=agg.values
            y=np.array([items[i][1] for i in keep])
            import re
            def _gid(v): 
                m=re.match(r"(id\d+)",v); return m.group(1) if m else "youtube_real"
            g=np.array([_gid(items[i][0]) for i in keep])
            n_dropped=len(items)-len(keep)
            if n_dropped: print(f"  celebdf: {n_dropped} item(s) had no frames and are excluded")
            pred=(s>=0.5).astype(int); lo,hi,sk,ng=grouped_ci(y,s,g)
            res["celebdf_zeroshot"]=dict(n_items_without_frames=int(n_dropped),
                n_duplicate_listings_removed=int(_ndup),
                population="full Celeb-DF v2 release, deduplicated by video id (DEV-008): NOT the "
                           "official 518-video test list, so NOT comparable with published benchmarks",
                auc=round(float(roc_auc_score(y,s)),4),
                grouped_ci=[round(lo,4),round(hi,4)] if lo is not None else None,
                macro_f1=round(float(f1_score(y,pred,average="macro")),4),
                mcc=round(float(matthews_corrcoef(y,pred)),4),
                n=int(len(y)),n_real=int((y==0).sum()),n_fake=int((y==1).sum()),n_groups=ng)
            for i,sc in zip(keep,s): rows.append(dict(video=items[i][0],label=items[i][1],p=float(sc),target="celebdf",dataset="CelebDF"))
            r=res["celebdf_zeroshot"]
            print(f"{'celebdf':16s} {r['auc']:7.4f} {str(r['grouped_ci']):>18s} {r['macro_f1']:7.4f} {r['mcc']:7.4f}  {r['n']}")
    else:
        res["celebdf_zeroshot"]={"status":"UNAVAILABLE","reason":"Celeb-DF-v2.json not built"}
        print("  celebdf: UNAVAILABLE - dataset json not built")
    pd.DataFrame(rows).to_csv(a.scores,index=False)
    json.dump(res,open(a.out,"w"),indent=1); print(f"\n-> {a.out}\n-> {a.scores}")

if __name__=="__main__": sys.exit(main() or 0)
