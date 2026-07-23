#!/usr/bin/env python3
"""EXP-5 Runtime / memory / complexity profiling (R2.2, R2.3, R3.6).
Profiles >=100 stratified videos (seed 42). Per-stage wall-clock timers wrap the frozen
extractor's stage functions (MediaPipe, optical flow, rPPG, frame-load, other). Peak RAM via
tracemalloc. RTF = processing_time / video_duration. Configs top-3/10/20/50 differ in classifier
inference + model size (extraction stages reused from the all-50 run; noted). Single-threaded
per video (workers=1) for clean timing. Xception inference profiled for the comparison table."""
import os,sys,json,time,tracemalloc,subprocess,datetime,random
import numpy as np, pandas as pd, warnings, cv2, psutil
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import precompute_features_best as P
from protocol import make_splits
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
SEED=42; random.seed(SEED); np.random.seed(SEED); F="features"; OUT="results_clean"
ROOT="/home/iiitn/Datasets/FaceForensics++"

# ---- wrap stage functions with cumulative timers ----
T={}
def wrap(mod,name,key):
    orig=getattr(mod,name)
    def w(*a,**k):
        t=time.perf_counter(); r=orig(*a,**k); T[key]=T.get(key,0.0)+(time.perf_counter()-t); return r
    setattr(mod,name,w); return orig
o1=wrap(P,"load_video_frames","frame_load"); o2=wrap(P,"get_landmarks","mediapipe")
o3=wrap(P,"extract_optical_flow","optical_flow"); o4=wrap(P,"extract_rppg","rppg")

# ---- select >=100 videos: 5 sources x 2 comps x 12, seed 42 (a few may fail to load) ----
specs=[("original_sequences/youtube","real"),("manipulated_sequences/Deepfakes","deepfakes"),
       ("manipulated_sequences/Face2Face","face2face"),("manipulated_sequences/FaceSwap","faceswap"),
       ("manipulated_sequences/NeuralTextures","neuraltextures")]
vids=[]
for sub,ds in specs:
    for comp in ("c23","c40"):
        d=f"{ROOT}/{sub}/{comp}/videos"
        fs=sorted([x for x in os.listdir(d) if x.endswith(".mp4")]); random.shuffle(fs)
        for fn in fs[:12]: vids.append((f"{d}/{fn}",ds,comp))
print(f"profiling {len(vids)} videos (5 sources x 2 comp x 12, seed 42)",flush=True)

proc=psutil.Process()
rows=[]
for vp,ds,comp in vids:
    cap=cv2.VideoCapture(vp); nfr=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 30; cap.release()
    dur=nfr/fps if fps>0 else np.nan
    T.clear(); tracemalloc.start()
    t0=time.perf_counter(); cpu0=proc.cpu_times()
    feat=P.process_single_video(vp,0,max_frames=300)
    total=time.perf_counter()-t0; cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    cpu1=proc.cpu_times(); cpu_used=(cpu1.user-cpu0.user)+(cpu1.system-cpu0.system)
    if feat is None: continue
    other=max(0.0,total-sum(T.get(k,0) for k in ("frame_load","mediapipe","optical_flow","rppg")))
    rows.append(dict(video=os.path.basename(vp),dataset=ds,comp=comp,duration_s=round(dur,2),frames=nfr,
        total_extract_s=round(total,3),frame_load_s=round(T.get("frame_load",0),3),mediapipe_s=round(T.get("mediapipe",0),3),
        optical_flow_s=round(T.get("optical_flow",0),3),rppg_s=round(T.get("rppg",0),3),other_s=round(other,3),
        peak_ram_mb=round(peak/1e6,1),cpu_time_s=round(cpu_used,2),rtf=round(total/dur,3) if dur>0 else None))
df=pd.DataFrame(rows); df.to_csv(f"{OUT}/runtime_per_video.csv",index=False)

# ---- classifier inference + model size per config (top-k) ----
# rank features by SHAP-importance proxy (LGBM gain) on train+val
def with_feats():
    real=make_splits(pd.read_csv(f"{F}/ffpp_original_c23.csv")); MAN=["deepfakes","face2face","faceswap","neuraltextures"]
    dfs=[real]+[make_splits(pd.read_csv(f"{F}/ffpp_{m}_c23.csv")) for m in MAN]
    d=pd.concat([x[x.partition.isin(["train","val"])] for x in dfs],ignore_index=True)
    FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")])
    for c in FC: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan); d[c]=d[c].fillna(d[c].median())
    return d,FC
dtr,FC=with_feats(); Xtr=StandardScaler().fit_transform(dtr[FC].values); ytr=dtr['label'].values.astype(int)
rank=lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1).fit(Xtr,ytr)
order=[FC[i] for i in np.argsort(rank.feature_importances_)[::-1]]
mean_extract=df.total_extract_s.mean(); mean_rtf=df.rtf.mean(); mean_ram=df.peak_ram_mb.mean()
cfgs=[]
for name,k in [("top-3",3),("top-10",10),("top-20",20),("all-50",50)]:
    cols=order[:k]; sc=StandardScaler().fit(dtr[cols].values)
    m=lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=1).fit(sc.transform(dtr[cols].values),ytr)
    X1=sc.transform(dtr[cols].values[:1])
    t=time.perf_counter()
    for _ in range(1000): m.predict_proba(X1)
    inf_ms=(time.perf_counter()-t)/1000*1000
    mfile=f"/tmp/_m{k}.txt"; m.booster_.save_model(mfile); sz=os.path.getsize(mfile)/1024; os.remove(mfile)
    cfgs.append(dict(config=name,n_features=k,extract_time_s=round(mean_extract,3),
        classifier_inference_ms=round(inf_ms,3),model_size_kb=round(sz,1),rtf=round(mean_rtf,3),peak_ram_mb=round(mean_ram,1)))

# ---- Xception inference cost (for comparison) ----
xcep={}
try:
    import torch, timm
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    xm=timm.create_model('legacy_xception',num_classes=1).to(dev).eval()
    x=torch.randn(8,3,299,299,device=dev)  # 8-frame video
    with torch.no_grad():
        for _ in range(3): xm(x)
        if dev=='cuda': torch.cuda.synchronize()
        t=time.perf_counter()
        for _ in range(20): xm(x)
        if dev=='cuda': torch.cuda.synchronize()
    xcep=dict(device=dev,per_video_8frame_ms=round((time.perf_counter()-t)/20*1000,2),
        model_size_mb=round(sum(p.numel() for p in xm.parameters())*4/1e6,1))
except Exception as e: xcep={"error":str(e)[:100]}

def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
hw=dict(cpu="ARM Cortex-X925 (NVIDIA GB10 SoC)",cores=20,arch="aarch64",ram_gb=121,gpu="NVIDIA GB10",os="Linux 6.17.0")
sw=dict(python="3.12.3",numpy="1.26.4",scipy="1.15.3",sklearn="1.7.2",lightgbm="4.6.0",opencv="4.11.0",mediapipe="0.10.18",torch="2.11.0+cu128",timm="1.0.28")
stage_means={k:round(df[k].mean(),3) for k in ["frame_load_s","mediapipe_s","optical_flow_s","rppg_s","other_s"]}
pd.DataFrame(cfgs).to_csv(f"{OUT}/runtime_profile.csv",index=False)
out=dict(provenance=dict(script="exp5_runtime.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    note="single-threaded per video (workers=1); extraction stages reused across configs; classifier inference measured per config"),
    hardware=hw,software=sw,n_videos=len(df),
    aggregate=dict(mean_total_extract_s=round(mean_extract,3),mean_rtf=round(mean_rtf,3),mean_peak_ram_mb=round(mean_ram,1),
        stage_means_s=stage_means),configs=cfgs,xception_inference=xcep)
json.dump(out,open(f"{OUT}/runtime.json","w"),indent=2)
print("\n=== EXP-5 RUNTIME PROFILE (n={} videos, CPU single-thread) ===".format(len(df)))
print(f"mean total extraction: {mean_extract:.2f}s/video | RTF: {mean_rtf:.3f} | peak RAM: {mean_ram:.0f}MB")
print(f"stage means (s): {stage_means}")
print(f"\n{'config':8s} {'n':>3s} {'extract_s':>10s} {'clf_inf_ms':>11s} {'model_kb':>9s} {'RTF':>6s}")
for c in cfgs: print(f"{c['config']:8s} {c['n_features']:3d} {c['extract_time_s']:10.2f} {c['classifier_inference_ms']:11.3f} {c['model_size_kb']:9.1f} {c['rtf']:6.3f}")
print(f"\nXception inference: {xcep}")
print(f"saved {OUT}/runtime_profile.csv, runtime_per_video.csv, runtime.json (commit {commit()})")
