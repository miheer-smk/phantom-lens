#!/usr/bin/env python3
"""G9 (Xception side) — persist Xception per-video predictions (FF++ test + Celeb-DF) and APPEND to
results_clean/predictions_per_video.csv (same schema). Then recompute PRISM-vs-Xception DeLong on
Celeb-DF FROM the persisted probs and compare to the locked value. GPU; run AFTER exp_g9_predictions.py.
"""
import os, sys, json, subprocess, datetime
import numpy as np, pandas as pd, warnings, cv2
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from delong import delong_roc_test
import torch, timm
from torch.utils.data import Dataset, DataLoader

SEED=42; F="features"; OUT="results_clean"; DEV='cuda' if torch.cuda.is_available() else 'cpu'
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
def basen(p): return os.path.basename(str(p))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
class DS(Dataset):
    def __init__(self,df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; im=cv2.imread(r.crop_path)
        if im is None: im=np.zeros((299,299,3),np.uint8)
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255.; im=(im-MEAN)/STD
        return torch.from_numpy(im.transpose(2,0,1)), i
def score(df):
    ps=np.zeros(len(df))
    with torch.no_grad():
        for x,idx in DataLoader(DS(df),batch_size=128,num_workers=8):
            ps[idx.numpy()]=torch.sigmoid(xm(x.to(DEV))).cpu().numpy().ravel()
    return ps

xm=timm.create_model('legacy_xception',num_classes=1)
xm.load_state_dict(torch.load("data_xception/xception_best.pt",map_location=DEV)); xm=xm.to(DEV).eval()

newrows=[]
def emit_video(vdf, dataset, manip, split, comp):
    # vdf: crops for ONE video with a 'p' column already; aggregate to video-level mean
    g=vdf.groupby(["video","label"])["p"].mean().reset_index()
    for _,r in g.iterrows():
        p=float(r["p"])
        newrows.append(dict(video_path=str(r["video"]), source_id=str(r["video"]).split("_")[0],
            dataset=dataset, manipulation=manip, compression=comp, true_label=int(r["label"]),
            pred_prob=round(p,6), pred_label=int(p>=0.5), split=split, model="Xception", seed=SEED))

# ---- FF++ test per manip ----
fm=pd.read_csv("data_xception/manifest_ffpp.csv").drop_duplicates("crop_path")
fm=fm[fm.split=="test"].copy(); fm["p"]=score(fm)
for ds,grp in fm.groupby("dataset"):
    emit_video(grp, "FFpp", ds, "test", "c23")

# ---- Celeb-DF ----
cm=pd.read_csv("data_xception/manifest_celebdf.csv").drop_duplicates("crop_path"); cm["p"]=score(cm)
emit_video(cm, "CelebDF", "NA", "zero_shot", "c23")

# ---- append to predictions_per_video.csv ----
pv=pd.read_csv(f"{OUT}/predictions_per_video.csv")
pv=pv[pv.model!="Xception"]  # idempotent re-run
out=pd.concat([pv, pd.DataFrame(newrows)], ignore_index=True)
out.to_csv(f"{OUT}/predictions_per_video.csv", index=False)

# ---- recompute PRISM-vs-Xception DeLong on Celeb-DF from persisted probs ----
prism=out[(out.model=="PRISM_50D_zeroshot")&(out.dataset=="CelebDF")][["video_path","true_label","pred_prob"]]
xcep =out[(out.model=="Xception")&(out.dataset=="CelebDF")][["video_path","pred_prob"]]
prism["vid"]=prism.video_path.map(lambda v: os.path.splitext(basen(v))[0])
xcep["vid"]=xcep.video_path.map(lambda v: os.path.splitext(basen(v))[0])
mg=prism.rename(columns={"pred_prob":"pP"}).merge(xcep.rename(columns={"pred_prob":"pX"}),on="vid")
aX,aP,z,p=delong_roc_test(mg.true_label.values.astype(int), mg.pX.values, mg.pP.values)
res=dict(comparison="PRISM_vs_Xception_CelebDF_from_persisted_predictions",n_matched=int(len(mg)),
    auc_xception=round(aX,4),auc_prism=round(aP,4),delta_xcep_minus_prism=round(aX-aP,4),z=round(z,3),p_value=float(p),
    locked_reference=dict(auc_xception=0.8211,auc_prism=0.6322,z=15.426))
json.dump(res,open(f"{OUT}/prism_vs_xception_from_predictions.json","w"),indent=1)

print("Xception per-video appended:",len(newrows),"rows |",", ".join(sorted(out.model.unique())))
print(f"PRISM-vs-Xception (Celeb-DF, from persisted probs): Xcep={aX:.4f} PRISM={aP:.4f} z={z:.3f} p={p:.2e}  (locked: 0.8211/0.6322/15.43)")
print(f"saved -> predictions_per_video.csv (+Xception), prism_vs_xception_from_predictions.json (commit {commit()})")
