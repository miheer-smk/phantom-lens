#!/usr/bin/env python
"""D2 / Phase 7.4 - Xception re-run with common aggregation and an INTACT join key.

BLOCKING ITEM. Closes DEFECT-002 by regenerating per-video Xception scores keyed on the real
video name, so the 6121-video intersection with PRISM can be rebuilt and the paired DeLong
z = 15.426 re-derived. Aggregation p_V = (1/m) sum_t p_t over the same sampled frames, identical
to what PRISM's comparison uses.
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import norm
R=Path.home()/"prism_r2"; LEG=R/"legacy"/"phantomlens"

def delong(y,p1,p2):
    y=np.asarray(y); pos=y==1; neg=~pos; m,n=pos.sum(),neg.sum()
    def comp(p):
        X,Y=p[pos],p[neg]
        v01=np.array([((Y<x).sum()+0.5*(Y==x).sum())/n for x in X])
        v10=np.array([((X>t).sum()+0.5*(X==t).sum())/m for t in Y])
        return v01,v10,v01.mean()
    a1,b1,A1=comp(p1); a2,b2,A2=comp(p2)
    S=np.cov(np.vstack([a1,a2]))/m + np.cov(np.vstack([b1,b2]))/n
    var=S[0,0]+S[1,1]-2*S[0,1]
    if var<=0: return A1,A2,np.nan,np.nan
    z=(A1-A2)/np.sqrt(var)
    return A1,A2,float(z),float(2*(1-norm.cdf(abs(z))))


def _training_normalisation():
    """Parse MEAN/STD out of the script the Xception checkpoint was trained with.

    Reading the value beats restating it: a restated constant can drift from the checkpoint
    silently, which is exactly the failure this guards against. If the training script cannot be
    read or parsed, this raises rather than falling back to a default - a wrong normalisation
    produces a plausible-looking wrong number, which is worse than a crash.
    """
    import ast, re
    train = (Path.home() / "prism_r2" / "legacy" / "phantomlens" /
             "Major Revision Results" / "00_logs" / "xception_train.py")
    if not train.exists():
        raise FileNotFoundError(
            f"cannot verify normalisation: training script not found at {train}. "
            "Refusing to guess - a wrong normalisation returns plausible scores silently.")
    txt = train.read_text()
    def grab(name):
        m = re.search(rf"{name}\s*=\s*np\.array\(\s*(\[[^\]]*\])", txt)
        if not m:
            raise ValueError(f"cannot parse {name} from {train.name}; refusing to guess")
        return np.array(ast.literal_eval(m.group(1)), np.float32)
    mean, std = grab("MEAN"), grab("STD")
    assert mean.shape == std.shape == (3,), f"unexpected normalisation shape {mean.shape}/{std.shape}"
    assert not np.allclose(std, 0), "STD contains zeros"
    print(f"normalisation read from {train.name}: MEAN={list(np.round(mean,4))} STD={list(np.round(std,4))}")
    return mean, std

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--out",default=str(R/"results"/"PD2_xception_rerun.json"))
    ap.add_argument("--scores",default=str(R/"results"/"xception_per_video_scores.csv"))
    ap.add_argument("--validate-only",action="store_true")
    a=ap.parse_args()
    import torch, timm, cv2
    from torch.utils.data import Dataset, DataLoader
    DEV="cuda" if torch.cuda.is_available() else "cpu"
    os.chdir(LEG)
    model=timm.create_model("legacy_xception",pretrained=False,num_classes=1)
    sd=torch.load("data_xception/xception_best.pt",map_location="cpu")
    if isinstance(sd,dict) and "state_dict" in sd: sd=sd["state_dict"]
    missing,unexpected=model.load_state_dict(sd,strict=False)
    print(f"checkpoint loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing or unexpected:
        print(f"  missing[:5]={list(missing)[:5]}\n  unexpected[:5]={list(unexpected)[:5]}")
    model.eval().to(DEV)
    # Normalisation is READ FROM THE TRAINING SCRIPT, not restated here. An earlier version of
    # this script hard-coded [0.5,0.5,0.5] while the checkpoint had been trained with ImageNet
    # constants. The model accepted the wrong inputs silently and returned plausible scores,
    # depressing every Table 19 AUC by 0.026-0.062 with no error and no warning. See DEV-010.
    MEAN, STD = _training_normalisation()
    class Crops(Dataset):
        def __init__(self,paths): self.p=paths
        def __len__(self): return len(self.p)
        def __getitem__(self,i):
            im=cv2.imread(self.p[i])
            if im is None: im=np.zeros((299,299,3),np.uint8)
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            if im.shape[:2]!=(299,299): im=cv2.resize(im,(299,299),interpolation=cv2.INTER_CUBIC)
            x=(im.astype(np.float32)/255.0-MEAN)/STD
            return torch.from_numpy(x.transpose(2,0,1)), i
    if a.validate_only:
        d=pd.read_csv("data_xception/manifest_celebdf.csv").drop_duplicates("crop_path").head(a.batch)
        dl=DataLoader(Crops(list(d.crop_path)),batch_size=a.batch,num_workers=2)
        with torch.no_grad():
            for xb,_ in dl:
                out=torch.sigmoid(model(xb.to(DEV))).squeeze(-1).cpu().numpy(); break
        print(f"VALIDATION OK: forward pass on {len(out)} crops, p range [{out.min():.4f},{out.max():.4f}]")
        return 0
    allv=[]
    for name,mf in (("CelebDF","data_xception/manifest_celebdf.csv"),
                    ("FFpp","data_xception/manifest_ffpp.csv")):
        d=pd.read_csv(mf).drop_duplicates("crop_path")
        d=d[d.crop_path.map(os.path.exists)]
        print(f"{name}: {len(d)} crops over {d.video.nunique()} videos",flush=True)
        dl=DataLoader(Crops(list(d.crop_path)),batch_size=a.batch,num_workers=4,pin_memory=True)
        ps=np.zeros(len(d),np.float32); t0=time.time()
        with torch.no_grad():
            for xb,idx in dl:
                ps[idx.numpy()]=torch.sigmoid(model(xb.to(DEV,non_blocking=True))).squeeze(-1).float().cpu().numpy()
        d=d.assign(p=ps)
        # COMMON AGGREGATION, join key INTACT
        v=d.groupby(["video","label","dataset"],as_index=False)["p"].mean()
        v["n_frames"]=d.groupby(["video","label","dataset"],as_index=False)["p"].size()["size"].values
        v["source"]=name
        allv.append(v)
        print(f"  scored in {(time.time()-t0)/60:.1f} min -> {len(v)} videos",flush=True)
    V=pd.concat(allv,ignore_index=True)
    V.to_csv(a.scores,index=False)
    from sklearn.metrics import roc_auc_score
    res={"generated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
         "aggregation":"p_V = (1/m) sum_t p_t over the sampled frames (common with PRISM)",
         "join_key":"real video name - DEFECT-002 closed","device":DEV}
    cd=V[V.source=="CelebDF"]
    res["xception_celebdf_full"]=dict(auc=round(float(roc_auc_score(cd.label,cd.p)),4),n=int(len(cd)),
                                      published=0.8207)
    # ---- the blocking check: shared 6121 with PRISM ----
    pr=pd.read_csv(LEG/"results_clean"/"predictions_per_video.csv")
    pr=pr[pr.model=="PRISM_50D_zeroshot"].copy()
    pr["vid"]=pr.video_path.map(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
    mg=pr.merge(cd[["video","p","label"]],left_on="vid",right_on="video",how="inner")
    print(f"\nshared videos with PRISM: {len(mg)} (published comparison used 6121)")
    if len(mg)>0:
        ax,ap_,z,pv=delong(mg.true_label.values.astype(int),mg.p.values,mg.pred_prob.values)
        res["shared_intersection"]=dict(n=int(len(mg)),n_expected=6121,
            auc_xception=round(float(ax),4),auc_prism=round(float(ap_),4),
            delong_z=round(float(z),3),p_value=float(pv),
            published_auc_xception=0.8211,published_auc_prism=0.6322,published_z=15.426,
            xception_matches=bool(abs(ax-0.8211)<5e-5),
            z_matches=bool(abs(z-15.426)<0.05))
        print(f"  Xception {ax:.4f} (published 0.8211)   PRISM {ap_:.4f} (published 0.6322)")
        print(f"  paired DeLong z = {z:.3f} (published 15.426)   p = {pv:.3g}")
        print(f"  -> Xception AUC matches: {res['shared_intersection']['xception_matches']}")
        print(f"  -> DeLong z matches    : {res['shared_intersection']['z_matches']}")
        if not res["shared_intersection"]["z_matches"]:
            print("  *** DEFECT: re-derived z does NOT match the published value - REPORT IMMEDIATELY ***")
    else:
        res["shared_intersection"]={"status":"UNAVAILABLE","reason":"no name overlap after re-scoring"}
    json.dump(res,open(a.out,"w"),indent=1)
    print(f"\n-> {a.out}\n-> {a.scores}")

if __name__=="__main__": sys.exit(main() or 0)
