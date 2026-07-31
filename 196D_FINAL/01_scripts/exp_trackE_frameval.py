import os,sys,json,re,subprocess,datetime; import numpy as np,pandas as pd,warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits; from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler; from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold; from sklearn.ensemble import RandomForestClassifier
CSV=sys.argv[1]; SEED=42; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
DIR={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else ("celebdf" if "Celeb-DF" in p else "?")
ev=pd.read_csv(CSV); ev["src"]=ev.video_path.map(method)
for c in FEATS: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); cd=ev[ev.src=="celebdf"].copy()
med=ff[ff.partition=="train"][FEATS].median(); ff[FEATS]=ff[FEATS].fillna(med); cd[FEATS]=cd[FEATS].fillna(med)
yc=cd.label.values.astype(int); ids=cd.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
tr=pd.concat([ff[(ff.src=='real')&(ff.partition=='train')].assign(label=0)]+[ff[(ff.src==m)&(ff.partition=='train')].assign(label=1) for m in MAN],ignore_index=True)
sc=StandardScaler().fit(tr[FEATS].values); m=RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1).fit(sc.transform(tr[FEATS].values),tr.label.values.astype(int))
p=m.predict_proba(sc.transform(cd[FEATS].values))[:,1]
a=[roc_auc_score(yc[i],p[i]) for _,i in GroupKFold(5).split(p,yc,ids) if len(np.unique(yc[i]))>1]
pr=(p>=0.5).astype(int); rr=float((pr[yc==0]==0).mean()); fr=float((pr[yc==1]==1).mean())
print(f"FULL {CSV}: celebdf_dev CV = {np.mean(a):.4f} ±{np.std(a):.3f} | realRec {rr:.3f} fakeRec {fr:.3f} (vs 60-frame R0 0.7018)")
json.dump(dict(csv=CSV,celebdf_dev_cv=[round(float(np.mean(a)),4),round(float(np.std(a)),4)],real_recall=round(rr,3),fake_recall=round(fr,3)),open("results_clean/trackE_100frame_dev.json","w"),indent=1)
