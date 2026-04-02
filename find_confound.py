import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DIM_NAMES = [
    'P1_vmr','P1_resStd','P1_hfRatio',
    'P2_prnu_E','P2_faceRatio',
    'P3_rgCorr','P3_bgCorr',
    'P4_faceBG','P4_specular','P4_shadow',
    'P5_driftM','P5_driftS',
    'P6_benford','P6_blockArt','P6_dblComp',
    'P7_resM','P7_resV',
    'P8_blurMag','P8_blurDir',
    'P9_flowMag',
    'P10_rgShift','P10_bgShift','P10_edgeCtr',
    'P11_eyeAsym','P11_tempStab','P11_geom',
    'P12_colorTemp','P12_fillLight','P12_satCons'
]

d   = pickle.load(open('data/precomputed_features_v3_base.pkl','rb'))
X   = np.array(d['features'])[:, [i for i in range(30) if i != 20]]
src = np.array(d['dataset_sources'])
lbl = np.array(d['labels'])

ffpp  = X[src == 'ffpp_official']
celeb = X[src == 'celebvhq']

print("="*65)
print("FEATURE-LEVEL DATASET CONFOUND ANALYSIS")
print("="*65)
print(f"FF++ samples   : {len(ffpp)}")
print(f"CelebVHQ samples: {len(celeb)}")
print()
print(f"{'Feature':<20} {'FF++ mean':>10} {'CelebVHQ mean':>14} {'Gap':>8} {'Confound?':>10}")
print("-"*65)

confounded = []
for i, name in enumerate(DIM_NAMES):
    fm  = ffpp[:, i].mean()
    cm  = celeb[:, i].mean()
    fs  = ffpp[:, i].std()
    cs  = celeb[:, i].std()
    gap = abs(fm - cm)
    pooled = np.sqrt((fs**2 + cs**2) / 2) + 1e-8
    d_val  = gap / pooled
    flag   = "YES ***" if d_val > 0.5 else "mild" if d_val > 0.2 else "no"
    if d_val > 0.5:
        confounded.append((name, d_val, fm, cm))
    print(f"{name:<20} {fm:>10.4f} {cm:>14.4f} {d_val:>8.3f} {flag:>10}")

print()
print("="*65)
print("MOST CONFOUNDED FEATURES (d > 0.5):")
print("="*65)
for name, d_val, fm, cm in sorted(confounded, key=lambda x: -x[1]):
    print(f"  {name:<20} d={d_val:.3f}  FF++={fm:.4f}  CelebVHQ={cm:.4f}")

print()
print("="*65)
print("SOLUTION:")
print("="*65)
print("These features differ between datasets for reasons UNRELATED")
print("to deepfake manipulation — they reflect codec/camera differences.")
print("Options:")
print("  1. Remove confounded features from training")
print("  2. Add CelebVHQ fake videos to training (fixes label imbalance)")
print("  3. Train separate scalers per dataset source")