"""
Phantom Lens — V2 Sequence Extraction (Overnight)
Extracts 8 frames × 24-dim per video → saves as sequence PKL
Safe: never touches original precomputed_features.pkl

Run TONIGHT: phantomlens_env\Scripts\python.exe extract_v2_sequences.py
Time: ~2-3 hours
Output: data/v2_sequences.pkl
"""

import os, sys, pickle, warnings
import cv2
import numpy as np
from scipy import fftpack
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── CONFIG ────────────────────────────────────────────────────────
FFPP_REAL_DIR  = r"data\ffpp_official\original_sequences\youtube"
FFPP_FAKE_BASE = r"data\ffpp_official\manipulated_sequences"
FAKE_TYPES     = ["Deepfakes", "Face2Face", "FaceShifter",
                  "FaceSwap", "NeuralTextures"]
OUT_PKL        = "data/v2_sequences.pkl"
CKPT_PKL       = "data/v2_sequences_checkpoint.pkl"
N_VIDEOS       = 800    # per class — enough for reliable AUC
N_FRAMES       = 8
SEED           = 42

# ── PHYSICS FUNCTIONS (V2) ────────────────────────────────────────
def compute_pillar1(img_gray):
    img = img_gray.astype(np.float32)
    h, w = img.shape
    vmr_vals = []
    for i in range(0, h-8, 8):
        for j in range(0, w-8, 8):
            block = img[i:i+8, j:j+8]
            mu = block.mean(); var = block.var()
            if mu > 5.0:
                vmr_vals.append(var/(mu+1e-6))
    vmr = float(np.median(vmr_vals)) if vmr_vals else 0.0
    blurred = cv2.medianBlur(img_gray, 3).astype(np.float32)
    residual_std = float((img-blurred).std())
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    hf_ratio = float(np.abs(lap).mean())/(float(np.abs(img).mean())+1e-6)
    return np.array([vmr, residual_std, hf_ratio], dtype=np.float32)

def compute_pillar2(img_gray):
    img = img_gray.astype(np.float32)
    h, w = img.shape
    blur = cv2.GaussianBlur(img,(5,5),1.0)
    prnu = img - blur
    cy,cx = h//2,w//2; h4,w4 = h//4,w//4
    face_e = float(np.abs(prnu[cy-h4:cy+h4,cx-w4:cx+w4]).mean())
    mask = np.ones((h,w),dtype=bool)
    mask[cy-h4:cy+h4,cx-w4:cx+w4] = False
    peri_e = float(np.abs(prnu[mask]).mean())
    return np.array([face_e, face_e/(peri_e+1e-6)], dtype=np.float32)

def compute_pillar3(img_rgb):
    img = img_rgb.astype(np.float32)
    def res(ch):
        b = cv2.GaussianBlur(ch.astype(np.uint8),(3,3),0).astype(np.float32)
        return (ch-b).flatten()
    r=res(img[:,:,0]); g=res(img[:,:,1]); b=res(img[:,:,2])
    def sc(a,b):
        if a.std()<1e-6 or b.std()<1e-6: return 0.0
        return float(np.corrcoef(a,b)[0,1])
    return np.array([sc(r,g),sc(b,g)], dtype=np.float32)

def compute_pillar4(img_rgb):
    h,w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    cy,cx = h//2,w//2; h4,w4 = h//4,w//4
    face = gray[cy-h4:cy+h4,cx-w4:cx+w4]
    top=gray[:h4,:]; left=gray[:,:w4]; right=gray[:,-w4:]
    diff = abs(float(face.mean())-
               float(np.concatenate([top.flatten(),
                                     left.flatten(),
                                     right.flatten()]).mean()))
    spec = float((face>0.9).mean())
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    shadow = float(np.sqrt(gx**2+gy**2).std())
    return np.array([diff,spec,shadow], dtype=np.float32)

def compute_pillar5_single(img_gray):
    img = img_gray.astype(np.float32)
    thr = np.percentile(img,95)
    mask = (img>thr).astype(np.float32)
    cnt = float(mask.mean())
    spread = float(np.argwhere(mask>0).std()) if mask.sum()>0 else 0.0
    return np.array([cnt,spread], dtype=np.float32)

def compute_pillar6(img_gray):
    img = img_gray.astype(np.float32)
    h,w = img.shape
    dct_c = []
    for i in range(0,h-8,8):
        for j in range(0,w-8,8):
            block = img[i:i+8,j:j+8]
            dc = fftpack.dct(fftpack.dct(block.T,norm='ortho').T,norm='ortho')
            dct_c.extend(np.abs(dc.flatten()[1:]))
    dct_c = np.array(dct_c); dct_c = dct_c[dct_c>1.0]
    if len(dct_c)>100:
        ld=np.floor(dct_c/10**np.floor(np.log10(dct_c+1e-10))).astype(int)
        ld=ld[(ld>=1)&(ld<=9)]
        obs=np.bincount(ld,minlength=10)[1:]/(len(ld)+1e-6)
        exp=np.array([np.log10(1+1/d) for d in range(1,10)])
        benford=float(np.sum(np.abs(obs-exp)))
    else: benford=0.0
    rows1=img[7::8,:]; rows2=img[8::8,:]
    mr=min(rows1.shape[0],rows2.shape[0])
    block_art=float(np.abs(rows1[:mr,:]-rows2[:mr,:]).mean()) if h>16 else 0.0
    hist,_=np.histogram(dct_c[:1000] if len(dct_c)>1000 else dct_c,
                        bins=50,range=(0,100))
    hf=np.abs(fftpack.fft(hist.astype(np.float32)))
    dcs=float(hf[2:8].max()/(hf[1:].mean()+1e-6))
    return np.array([benford,block_art,dcs], dtype=np.float32)

def compute_pillar7(frames_gray):
    if len(frames_gray)<2: return np.array([0.0,0.0],dtype=np.float32)
    residuals=[]; spatial_vars=[]
    for i in range(1,len(frames_gray)):
        prev=cv2.resize(frames_gray[i-1],(224,224)).astype(np.float32)
        curr=cv2.resize(frames_gray[i],(224,224)).astype(np.float32)
        res=np.abs(curr-prev)
        residuals.append(float(res.mean()))
        h,w=res.shape; cy,cx=h//2,w//2; h4,w4=h//4,w//4
        spatial_vars.append(float(res[cy-h4:cy+h4,cx-w4:cx+w4].var()))
    return np.array([float(np.mean(residuals)),
                     float(np.mean(spatial_vars))],dtype=np.float32)

def compute_pillar8(frames_gray):
    if len(frames_gray)<2:
        img=frames_gray[0].astype(np.float32)
        gx=cv2.Sobel(img,cv2.CV_64F,1,0)
        gy=cv2.Sobel(img,cv2.CV_64F,0,1)
        mag=float(np.sqrt(gx**2+gy**2).mean())
        angles=np.arctan2(np.abs(gy)+1e-6,np.abs(gx)+1e-6)
        dc=float(1.0/(angles.std()+1e-6))
        return np.array([mag,min(dc,10.0)],dtype=np.float32)
    mags=[]; dirs=[]
    for i in range(1,len(frames_gray)):
        diff=frames_gray[i].astype(np.float32)-frames_gray[i-1].astype(np.float32)
        gx=cv2.Sobel(diff,cv2.CV_64F,1,0); gy=cv2.Sobel(diff,cv2.CV_64F,0,1)
        mags.append(float(np.sqrt(gx**2+gy**2).mean()))
        dirs.append(float(np.arctan2(np.abs(gy).mean(),np.abs(gx).mean()+1e-6)))
    return np.array([float(np.mean(mags)),
                     min(float(1.0/(np.std(dirs)+1e-6)),100.0)],dtype=np.float32)

def compute_pillar9(frames_gray):
    if len(frames_gray)<2: return np.array([0.0,0.0],dtype=np.float32)
    fmags=[]; bdiscs=[]
    for i in range(1,min(len(frames_gray),4)):
        try:
            flow=cv2.calcOpticalFlowFarneback(
                frames_gray[i-1],frames_gray[i],None,
                0.5,2,10,2,5,1.1,0)
            mag=float(np.sqrt(flow[...,0]**2+flow[...,1]**2).mean())
            fmags.append(mag)
            fmm=np.sqrt(flow[...,0]**2+flow[...,1]**2)
            fg=cv2.Laplacian(fmm.astype(np.float32),cv2.CV_64F)
            bdiscs.append(float(np.abs(fg).mean()))
        except: fmags.append(0.0); bdiscs.append(0.0)
    return np.array([float(np.mean(fmags)),
                     float(np.mean(bdiscs))],dtype=np.float32)

def compute_pillar10(img_rgb):
    img=img_rgb.astype(np.float32)
    r=img[:,:,0]; g=img[:,:,1]; b=img[:,:,2]
    h,w=g.shape
    def cs(c1,c2):
        try:
            f1=np.fft.fft2(c1); f2=np.fft.fft2(c2)
            cp=f1*np.conj(f2); cp/=(np.abs(cp)+1e-10)
            corr=np.abs(np.fft.ifft2(cp))
            pk=np.unravel_index(corr.argmax(),corr.shape)
            dy=pk[0] if pk[0]<h//2 else pk[0]-h
            dx=pk[1] if pk[1]<w//2 else pk[1]-w
            return float(np.sqrt(dy**2+dx**2))
        except: return 0.0
    em=max(h//8,1)
    cr=float(np.abs(r[em:-em,em:-em]-g[em:-em,em:-em]).mean())
    er=float(np.abs(r-g).mean())
    return np.array([cs(r,g),cs(b,g),er/(cr+1e-6)],dtype=np.float32)

def extract_frame_24dim(frame_bgr, p7, p8, p9, p5):
    """Extract 24-dim feature from one frame with shared temporal features."""
    try:
        frame_rgb  = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY)
        frame_rgb  = cv2.resize(frame_rgb, (224,224))
        frame_gray = cv2.resize(frame_gray,(224,224))
        p1  = compute_pillar1(frame_gray)
        p2  = compute_pillar2(frame_gray)
        p3  = compute_pillar3(frame_rgb)
        p4  = compute_pillar4(frame_rgb)
        p6  = compute_pillar6(frame_gray)
        p10 = compute_pillar10(frame_rgb)
        feat = np.concatenate([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10])
        feat = np.nan_to_num(feat,nan=0.0,posinf=0.0,neginf=0.0)
        return feat.astype(np.float32)
    except: return None

def extract_sequence(video_path, n_frames=N_FRAMES):
    """Returns (n_frames, 24) array or None."""
    try:
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2: total = 30
        start = int(total*0.20); end = int(total*0.80)
        if end-start < n_frames: start=0; end=total
        indices = [int(x) for x in np.linspace(start,end-1,n_frames)]
        frames_bgr=[]; frames_gray=[]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ret,frame=cap.read()
            if ret and frame is not None:
                f=cv2.resize(frame,(224,224))
                frames_bgr.append(f)
                frames_gray.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY))
        cap.release()
        if len(frames_bgr)==0: return None

        # Temporal features computed once per video
        p7=compute_pillar7(frames_gray)
        p8=compute_pillar8(frames_gray)
        p9=compute_pillar9(frames_gray)
        p5=compute_pillar5_single(frames_gray[0])

        seq=[]
        for fb in frames_bgr:
            feat=extract_frame_24dim(fb,p7,p8,p9,p5)
            if feat is not None: seq.append(feat)

        if len(seq)==0: return None
        # Pad or trim to exactly n_frames
        while len(seq)<n_frames: seq.append(seq[-1])
        seq=seq[:n_frames]
        return np.array(seq,dtype=np.float32)  # (8, 24)
    except: return None

def get_videos(folder, max_n, seed=SEED):
    exts=('.mp4','.avi','.mov','.mkv')
    videos=[]
    for root,_,files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(exts):
                videos.append(os.path.join(root,f))
    rng=np.random.default_rng(seed)
    if len(videos)>max_n:
        idx=rng.choice(len(videos),max_n,replace=False)
        videos=[videos[i] for i in sorted(idx)]
    return videos[:max_n]

# ── LOAD CHECKPOINT IF EXISTS ─────────────────────────────────────
if os.path.exists(CKPT_PKL):
    print(f"Resuming from checkpoint: {CKPT_PKL}")
    with open(CKPT_PKL,'rb') as f: ckpt=pickle.load(f)
    sequences  = ckpt['sequences']
    labels     = ckpt['labels']
    sources    = ckpt['sources']
    gen_types  = ckpt['gen_types']
    done_types = ckpt['done_types']
    print(f"  Already done: {done_types}")
    print(f"  Sequences so far: {len(sequences)}")
else:
    sequences=[]; labels=[]; sources=[]; gen_types=[]; done_types=[]

def save_checkpoint():
    with open(CKPT_PKL,'wb') as f:
        pickle.dump({'sequences':sequences,'labels':labels,
                     'sources':sources,'gen_types':gen_types,
                     'done_types':done_types},f,protocol=4)
    print(f"  Checkpoint saved ({len(sequences)} sequences)")

# ── EXTRACT ───────────────────────────────────────────────────────
print("="*60)
print("V2 SEQUENCE EXTRACTION")
print(f"  {N_VIDEOS} videos per class × {N_FRAMES} frames = sequences")
print("="*60)

# Real
if 'real' not in done_types:
    print(f"\n[Real] {FFPP_REAL_DIR}")
    videos=get_videos(FFPP_REAL_DIR,N_VIDEOS)
    print(f"  Found {len(videos)} videos")
    ok=0
    for i,vp in enumerate(videos):
        seq=extract_sequence(vp)
        if seq is not None:
            sequences.append(seq); labels.append(0)
            sources.append('ffpp_official'); gen_types.append('real')
            ok+=1
        if (i+1)%50==0:
            print(f"  {i+1}/{len(videos)} | ok={ok}")
            save_checkpoint()
    done_types.append('real')
    save_checkpoint()
    print(f"  Real done: {ok} sequences")

# Fakes
for ftype in FAKE_TYPES:
    if ftype.lower() in done_types:
        print(f"  [{ftype}] already done — skipping")
        continue
    fdir=os.path.join(FFPP_FAKE_BASE,ftype)
    if not os.path.exists(fdir):
        print(f"  [{ftype}] not found — skipping"); continue
    print(f"\n[{ftype}] {fdir}")
    videos=get_videos(fdir,N_VIDEOS)
    print(f"  Found {len(videos)} videos")
    ok=0
    for i,vp in enumerate(videos):
        seq=extract_sequence(vp)
        if seq is not None:
            sequences.append(seq); labels.append(1)
            sources.append('ffpp_official'); gen_types.append(ftype.lower())
            ok+=1
        if (i+1)%50==0:
            print(f"  {i+1}/{len(videos)} | ok={ok}")
            save_checkpoint()
    done_types.append(ftype.lower())
    save_checkpoint()
    print(f"  {ftype} done: {ok} sequences")

# ── SAVE FINAL PKL ────────────────────────────────────────────────
print("\nSaving final PKL...")
final={
    'sequences': np.array(sequences,dtype=np.float32),  # (N,8,24)
    'labels':    np.array(labels,   dtype=np.int32),
    'sources':   np.array(sources),
    'gen_types': np.array(gen_types),
}
with open(OUT_PKL,'wb') as f: pickle.dump(final,f,protocol=4)

N=len(labels)
print(f"\nDone! Saved to {OUT_PKL}")
print(f"  Shape: {final['sequences'].shape}")
print(f"  Total: {N} | Real: {int((final['labels']==0).sum())} "
      f"| Fake: {int((final['labels']==1).sum())}")
print(f"\nNext step — run:")
print(f"  phantomlens_env\\Scripts\\python.exe train_v2_bilstm.py")