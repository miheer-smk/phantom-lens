#!/usr/bin/env python3
"""EXP-10 signal traces for the 4 case videos (R4 visible evidence).
Per case, extract the per-frame physical signals underlying the top SHAP features:
 - mouth-region texture stability (frame-to-frame correlation) -> underlies roi_mouth_texture_flicker
 - eye-aspect-ratio / blink trace -> underlies t_blink_symmetry
Plot both. Caption states SHAP=explanation not causation."""
import os,sys
import numpy as np, cv2
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"src")
import precompute_features_best as P
import roi_config as RC
CAVEAT="Signals underlie the SHAP-attributed features; SHAP explains the classifier's output, not causation."
FIG="Major Revision Results/03_figures/exp10_case_level_shap"
CASES={
 "tp":("/home/iiitn/Datasets/FaceForensics++/manipulated_sequences/Deepfakes/c23/videos/739_865.mp4","TP Deepfakes (fake, P=0.9996)"),
 "tn":("/home/iiitn/Datasets/FaceForensics++/original_sequences/youtube/c23/videos/949.mp4","TN real (P=0.022)"),
 "fn":("/home/iiitn/Datasets/FaceForensics++/manipulated_sequences/Face2Face/c23/videos/128_896.mp4","FN Face2Face (fake missed, P=0.078)"),
 "fp":("/home/iiitn/Datasets/Celeb-DF-v2/YouTube-real/00111.mp4","FP CelebDF real (P=0.993 -> flagged fake)"),
}
def signals(vpath):
    frames,fps=P.load_video_frames(vpath,max_frames=150)
    fm=P.init_face_mesh(); mouth=[]; ear=[]; prev=None
    for f in frames:
        rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB); lm=P.get_landmarks(fm,rgb)
        if lm is None: mouth.append(np.nan); ear.append(np.nan); continue
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        m=P.landmarks_to_mask(lm,RC.MOUTH_REGION,g.shape); ys,xs=np.where(m>0)
        if len(xs)>30:
            patch=cv2.resize(g[ys.min():ys.max()+1,xs.min():xs.max()+1].astype(np.float32),(24,24)).ravel()
            if prev is not None and patch.std()>1e-6 and prev.std()>1e-6:
                mouth.append(float(np.corrcoef(patch,prev)[0,1]))
            else: mouth.append(np.nan)
            prev=patch
        else: mouth.append(np.nan)
        try:
            l=P.compute_ear(lm,P.LEFT_EYE); r=P.compute_ear(lm,P.RIGHT_EYE); ear.append((l+r)/2)
        except Exception: ear.append(np.nan)
    fm.close(); return np.array(mouth),np.array(ear)

fig,axes=plt.subplots(4,2,figsize=(12,11))
for i,(tag,(vp,title)) in enumerate(CASES.items()):
    mouth,ear=signals(vp)
    ax=axes[i,0]; ax.plot(mouth,color="#c0392b",lw=1); ax.axhline(np.nanmean(mouth),ls="--",c="gray",lw=0.8)
    ax.set_title(f"{title}\nmouth-region frame-to-frame texture corr (higher=stable)",fontsize=8)
    ax.set_ylim(-0.2,1.05); ax.set_ylabel("corr"); ax.set_xlabel("frame")
    ax2=axes[i,1]; ax2.plot(ear,color="#2471a3",lw=1)
    ax2.set_title("eye-aspect-ratio (blink trace)",fontsize=8); ax2.set_xlabel("frame"); ax2.set_ylabel("EAR")
plt.figtext(0.5,0.005,CAVEAT,ha="center",fontsize=7)
plt.tight_layout(rect=[0,0.02,1,1]); plt.savefig(f"{FIG}/case_signals_all.png",dpi=140,bbox_inches="tight"); plt.close()
print(f"saved {FIG}/case_signals_all.png")
# also per-case individual signal plot
for tag,(vp,title) in CASES.items():
    mouth,ear=signals(vp)
    fig,ax=plt.subplots(1,2,figsize=(9,3))
    ax[0].plot(mouth,color="#c0392b",lw=1.2); ax[0].set_title("mouth texture stability",fontsize=8); ax[0].set_ylim(-0.2,1.05)
    ax[1].plot(ear,color="#2471a3",lw=1.2); ax[1].set_title("blink (EAR)",fontsize=8)
    fig.suptitle(title,fontsize=9); plt.figtext(0.5,-0.03,CAVEAT,ha="center",fontsize=6)
    plt.tight_layout(); plt.savefig(f"{FIG}/case_signal_{tag}.png",dpi=130,bbox_inches="tight"); plt.close()
    print(f"  saved case_signal_{tag}.png")
