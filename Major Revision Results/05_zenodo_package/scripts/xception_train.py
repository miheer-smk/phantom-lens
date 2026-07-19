#!/usr/bin/env python3
"""Xception baseline training + eval (fair DL comparison under identity-disjoint splits).
Train on FF++ c23 train identities (real+4 manips), select by val AUC, evaluate video-level on:
FF++ test identities (per-manip + overall) and Celeb-DF v2 zero-shot. Same protocol as PRISM."""
import os,sys,json,time,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm, cv2
from sklearn.metrics import roc_auc_score, recall_score
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)

ff=pd.read_csv("data_xception/manifest_ffpp.csv").drop_duplicates("crop_path")
cd=pd.read_csv("data_xception/manifest_celebdf.csv").drop_duplicates("crop_path")
class DS(Dataset):
    def __init__(self,df,train=False): self.df=df.reset_index(drop=True); self.train=train
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; im=cv2.imread(r.crop_path)
        if im is None: im=np.zeros((299,299,3),np.uint8)
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        if self.train and np.random.rand()<0.5: im=im[:,::-1,:].copy()
        im=(im-MEAN)/STD
        return torch.from_numpy(im.transpose(2,0,1)), np.float32(r.label), i

def loaders():
    tr=ff[ff.split=="train"]; va=ff[ff.split=="val"]
    return (DataLoader(DS(tr,True),batch_size=64,shuffle=True,num_workers=8,drop_last=True),
            DataLoader(DS(va),batch_size=128,num_workers=8), va)

def epoch_train(model,dl,opt,lossf):
    model.train()
    for x,y,_ in dl:
        x,y=x.to(DEV),y.to(DEV).unsqueeze(1)
        opt.zero_grad(); out=model(x); loss=lossf(out,y); loss.backward(); opt.step()

@torch.no_grad()
def predict(model,df):
    model.eval(); dl=DataLoader(DS(df),batch_size=128,num_workers=8); ps=np.zeros(len(df))
    for x,_,idx in dl:
        p=torch.sigmoid(model(x.to(DEV))).cpu().numpy().ravel(); ps[idx.numpy()]=p
    d=df.copy(); d["p"]=ps
    vid=d.groupby(["video","dataset","label"])["p"].mean().reset_index()  # video-level aggregation
    return vid

def auc_recall(vid):
    y=vid.label.values; p=vid.p.values
    return (roc_auc_score(y,p), recall_score(y,(p>=.5).astype(int),pos_label=0),
            recall_score(y,(p>=.5).astype(int),pos_label=1))

def main():
    print(f"[env] torch {torch.__version__} dev={DEV} {torch.cuda.get_device_name(0) if DEV=='cuda' else ''}",flush=True)
    tr_dl,va_dl,va=loaders()
    model=timm.create_model('legacy_xception',pretrained=True,num_classes=1).to(DEV)
    # class balance via pos_weight
    n_pos=(ff[ff.split=="train"].label==1).sum(); n_neg=(ff[ff.split=="train"].label==0).sum()
    lossf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg/max(n_pos,1)]).to(DEV))
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    best=-1; best_state=None
    for ep in range(8):
        t=time.time(); epoch_train(model,tr_dl,opt,lossf)
        vid=predict(model,va); a,_,_=auc_recall(vid)
        print(f"  epoch {ep+1}/8  val video-AUC={a:.4f}  ({time.time()-t:.0f}s)",flush=True)
        if a>best: best=a; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # ---- eval ----
    res={"provenance":{"script":"xception_train.py","git_commit":subprocess.getoutput("git rev-parse --short HEAD"),
        "seed":SEED,"date":datetime.date.today().isoformat(),"hardware":torch.cuda.get_device_name(0),
        "model":"legacy_xception (timm, ImageNet-pretrained)","protocol":"identity-disjoint FF++; video-level mean aggregation; CelebDF zero-shot"},
        "val_best_auc":round(best,4)}
    # FF++ test overall + per-manip
    fftest=ff[ff.split=="test"]; vid=predict(model,fftest)
    a,rr,fr=auc_recall(vid); res["ffpp_test_overall"]={"auc":round(a,4),"real_recall":round(rr,4),"fake_recall":round(fr,4)}
    perm={}
    for ds in ["deepfakes","face2face","faceswap","neuraltextures"]:
        sub=vid[(vid.dataset==ds)|(vid.dataset=="real")]
        if sub.label.nunique()>1: perm[ds]=round(roc_auc_score(sub.label,sub.p),4)
    res["ffpp_per_manip_auc"]=perm
    # CelebDF zero-shot
    t0=time.time(); vidc=predict(model,cd); inf=time.time()-t0
    ac,rrc,frc=auc_recall(vidc)
    res["celebdf_zeroshot"]={"auc":round(ac,4),"real_recall":round(rrc,4),"fake_recall":round(frc,4),
        "n_real":int((vidc.label==0).sum()),"n_fake":int((vidc.label==1).sum())}
    res["inference_sec_celebdf"]=round(inf,1)
    res["model_size_mb"]=round(sum(p.numel() for p in model.parameters())*4/1e6,1)
    torch.save(best_state,"data_xception/xception_best.pt")
    json.dump(res,open("results_clean/xception_baseline.json","w"),indent=2)
    print("\n=== XCEPTION BASELINE (identity-disjoint) ===")
    print(f"  FF++ test overall AUC = {res['ffpp_test_overall']['auc']}  per-manip={perm}")
    print(f"  CelebDF zero-shot AUC = {res['celebdf_zeroshot']['auc']}  real_rec={rrc:.3f} fake_rec={frc:.3f}")
    print(f"  model {res['model_size_mb']}MB, hardware {res['provenance']['hardware']}")
    print("saved results_clean/xception_baseline.json")

if __name__=="__main__": main()
