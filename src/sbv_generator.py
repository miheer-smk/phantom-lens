#!/usr/bin/env python3
"""Track E3 — Self-Blended VIDEO generator (temporal adaptation of SBI).
For a REAL video, synthesise a fake by blending each frame with a transformed copy of ITSELF
(source) under a soft landmark-hull mask (feathered boundary). No real deepfakes used -> forces the
detector onto the universal compositing/boundary artifact rather than FF++-specific signatures.
Novel part: TEMPORAL artifact injection (per-frame boundary jitter, STG-param flicker, single-frame
outliers, landmark perturbation) so the 37 temporal features see the inconsistency real deepfakes show.
Deterministic per (video, seed, jitter). Returns blended BGR frames + the mask used (for boundary feats).
"""
import numpy as np, cv2

def _rand_affine(shape, rng, strength):
    h,w=shape[:2]; d=strength*0.03
    src=np.float32([[0,0],[w,0],[0,h]])
    dst=src+rng.uniform(-d,d,src.shape).astype(np.float32)*np.float32([w,h])
    return cv2.getAffineTransform(src,dst)

def stg_source(frame, rng, strength=1.0):
    """Source-target generator: a transformed copy of the frame (compositing mismatch)."""
    img=frame.astype(np.float32); ops=rng.permutation(7)
    for op in ops:
        if op==0:  # per-channel colour gain/offset
            img=img*rng.uniform(1-0.1*strength,1+0.1*strength,3)+rng.uniform(-8*strength,8*strength,3)
        elif op==1 and rng.rand()<0.5:  # gaussian blur
            k=int(rng.choice([3,5])); img=cv2.GaussianBlur(img,(k,k),0)
        elif op==2 and rng.rand()<0.4:  # sharpen
            img=cv2.filter2D(img,-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32))
        elif op==3 and rng.rand()<0.5:  # resolution mismatch (down->up)
            h,w=img.shape[:2]; s=rng.uniform(0.5,0.9)
            img=cv2.resize(cv2.resize(img,(max(int(w*s),8),max(int(h*s),8))),(w,h))
        elif op==4 and rng.rand()<0.5:  # JPEG re-encode at different quality
            q=int(rng.uniform(40,90)); ok,enc=cv2.imencode('.jpg',np.clip(img,0,255).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,q])
            if ok: img=cv2.imdecode(enc,cv2.IMREAD_COLOR).astype(np.float32)
        elif op==5:  # brightness/contrast
            img=img*rng.uniform(1-0.15*strength,1+0.15*strength)+rng.uniform(-12*strength,12*strength)
        elif op==6 and rng.rand()<0.4:  # slight affine warp
            h,w=img.shape[:2]; M=_rand_affine(img.shape,rng,strength); img=cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_REFLECT)
    return np.clip(img,0,255).astype(np.uint8)

def soft_mask(landmarks, shape, rng, jitter_px=0.0):
    """Feathered convex-hull face mask with elastic deform + erosion + two-pass gaussian."""
    h,w=shape[:2]; m=np.zeros((h,w),np.uint8)
    pts=landmarks[:,:2].astype(np.int32)
    if jitter_px>0: pts=pts+rng.uniform(-jitter_px,jitter_px,pts.shape).astype(np.int32)  # per-frame boundary jitter
    hull=cv2.convexHull(pts); cv2.fillConvexPoly(m,hull,255)
    k=max(int(0.04*min(h,w)),3); m=cv2.erode(m,np.ones((k,k),np.uint8))
    m=cv2.GaussianBlur(m,(0,0),sigmaX=k*0.6); m=cv2.GaussianBlur(m,(0,0),sigmaX=k*0.3)
    return (m.astype(np.float32)/255.0)[...,None]

class SBVGenerator:
    """Per-video temporal self-blend. temporal_jitter scales the per-frame inconsistency."""
    def __init__(self, seed=42, temporal_jitter=1.0):
        self.rng=np.random.RandomState(seed); self.tj=temporal_jitter
        self.base_strength=self.rng.uniform(0.6,1.2)   # per-video base compositing strength

    def frame(self, frame_bgr, landmarks):
        if landmarks is None: return frame_bgr, None
        rng=self.rng
        # STG-param flicker: per-frame strength perturbation + occasional single-frame outlier
        s=self.base_strength*(1+rng.uniform(-0.25,0.25)*self.tj)
        if rng.rand()<0.05*self.tj: s*=rng.uniform(1.5,2.5)          # single-frame outlier
        lm=landmarks.copy()
        lm[:,:2]+=rng.uniform(-1.5,1.5,lm[:,:2].shape)*self.tj        # per-frame landmark perturbation
        source=stg_source(frame_bgr,rng,strength=s)
        mask=soft_mask(lm,frame_bgr.shape,rng,jitter_px=2.0*self.tj)  # per-frame boundary jitter
        blended=(source.astype(np.float32)*mask+frame_bgr.astype(np.float32)*(1-mask))
        return np.clip(blended,0,255).astype(np.uint8), mask

if __name__=="__main__":  # smoke: measure blend boundary artifact strength
    import sys; sys.path.insert(0,"src"); import precompute_features_best as P
    frames,fps=P.load_video_frames(sys.argv[1] if len(sys.argv)>1 else
        "/home/iiitn/Datasets/FaceForensics++/original_sequences/youtube/c23/videos/000.mp4",max_frames=8)
    fm=P.init_face_mesh(); g=SBVGenerator(seed=42,temporal_jitter=1.0); diffs=[]
    for f in frames:
        lm=P.get_landmarks(fm,cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
        b,m=g.frame(f,lm)
        if m is not None: diffs.append(float(np.mean(np.abs(b.astype(float)-f.astype(float)))))
    fm.close(); print(f"SBV smoke: {len(diffs)} frames blended, mean|blended-real|={np.mean(diffs):.2f} (nonzero -> blend active)")
