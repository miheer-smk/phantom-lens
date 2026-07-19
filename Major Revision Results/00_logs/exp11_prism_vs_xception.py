#!/usr/bin/env python3
"""EXP-11 addendum — DeLong PRISM vs Xception on CelebDF zero-shot (paired, same videos).
Xception: re-score saved xception_best.pt on CelebDF crops -> video-level. PRISM: 50-D on CelebDF.
Align by video id, DeLong. Also FF++ test if feasible."""
import os,sys,json,subprocess,datetime
import numpy as np, pandas as pd, warnings, cv2
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits
from delong import delong_roc_test
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, torch, timm
from torch.utils.data import Dataset, DataLoader
SEED=42; F="features"; OUT="results_clean"; DEV='cuda' if torch.cuda.is_available() else 'cpu'
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def base(p): return os.path.basename(str(p))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
# ---- Xception CelebDF video-level scores ----
cd_manifest=pd.read_csv("data_xception/manifest_celebdf.csv").drop_duplicates("crop_path")
class DS(Dataset):
    def __init__(self,df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; im=cv2.imread(r.crop_path)
        if im is None: im=np.zeros((299,299,3),np.uint8)
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255.; im=(im-MEAN)/STD
        return torch.from_numpy(im.transpose(2,0,1)), i
xm=timm.create_model('legacy_xception',num_classes=1); xm.load_state_dict(torch.load("data_xception/xception_best.pt",map_location=DEV)); xm=xm.to(DEV).eval()
ps=np.zeros(len(cd_manifest))
with torch.no_grad():
    for x,idx in DataLoader(DS(cd_manifest),batch_size=128,num_workers=8):
        ps[idx.numpy()]=torch.sigmoid(xm(x.to(DEV))).cpu().numpy().ravel()
cd_manifest["p"]=ps
xvid=cd_manifest.groupby(["video","label"])["p"].mean().reset_index(); xvid["vid"]=xvid.video.map(lambda v: str(v))
# ---- PRISM 50-D CelebDF video-level scores ----
def load(name):
    return make_splits(pd.read_csv(f"{F}/ffpp_{'original' if name=='real' else name}_c23.csv"))
real=load("real"); MANd={m:load(m) for m in MAN}
FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")])
def clean(df):
    d=df.copy()
    for c in FC: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan); d[c]=d[c].fillna(d[c].median())
    return d
real=clean(real); MANd={m:clean(v) for m,v in MANd.items()}
tr=pd.concat([real[real.partition=="train"]]+[MANd[m][MANd[m].partition=="train"] for m in MAN],ignore_index=True)
cd=clean(pd.read_csv(f"{F}/celebdf_features.csv"))
sc=StandardScaler().fit(tr[FC].values); clf=lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1).fit(sc.transform(tr[FC].values),tr['label'].values.astype(int))
cd["p"]=clf.predict_proba(sc.transform(cd[FC].values))[:,1]; cd["vid"]=cd.video_path.map(lambda p: os.path.splitext(base(p))[0])
# ---- align by video id (basename without extension), DeLong ----
xvid["vid"]=xvid.vid.map(lambda v: os.path.splitext(str(v))[0])
mg=cd[["vid","label","p"]].rename(columns={"p":"p_prism"}).merge(xvid[["vid","p"]].rename(columns={"p":"p_xcep"}),on="vid")
y=mg.label.values.astype(int); pP=mg.p_prism.values; pX=mg.p_xcep.values
aP,aX,z,p=delong_roc_test(y,pX,pP)  # test Xception vs PRISM
res=dict(comparison="PRISM_vs_Xception_CelebDF",test="DeLong",n_matched_videos=int(len(mg)),
    auc_xception=round(aX,4),auc_prism=round(aP,4),auc_diff_xcep_minus_prism=round(aX-aP,4),z=round(z,3),p_value=float(p))
# merge into statistical_tests.json
st=json.load(open(f"{OUT}/statistical_tests.json")) if os.path.exists(f"{OUT}/statistical_tests.json") else {"delong":[]}
st.setdefault("prism_vs_xception",[]).append(res)
json.dump(st,open(f"{OUT}/statistical_tests.json","w"),indent=2)
print("=== PRISM vs Xception (CelebDF, DeLong paired) ===")
print(f"  matched videos: {res['n_matched_videos']}")
print(f"  Xception AUC={aX:.4f}  PRISM AUC={aP:.4f}  Δ(X-P)={aX-aP:+.4f}  z={z:+.3f}  p={p:.3e}")
print(f"saved -> {OUT}/statistical_tests.json (commit {commit()})")
