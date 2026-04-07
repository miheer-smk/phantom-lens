import pickle
import numpy as np

with open('data/precomputed_features.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys:", list(data.keys()))
print("Features shape:", data['features'].shape)
print("Labels:", np.unique(data['labels'], return_counts=True))
print("Sources:", sorted(set(data['dataset_sources'])))
print("Generators:", sorted(set(data['generator_types'])))

feats = data['features']
names = [
    'P1_vmr','P1_resStd','P1_hfRatio',
    'P2_energy','P2_faceRatio',
    'P3_rgCorr','P3_bgCorr',
    'P4_faceBG','P4_specRatio','P4_shadow',
    'P5_driftMean','P5_driftStd',
    'P6_benford','P6_blockArt','P6_dblComp',
    'P7_resMean','P7_resVar',
    'P8_blurMag','P8_dirConsist',
    'P9_flowMag','P9_boundary',
    'P10_rgShift','P10_bgShift','P10_edgeRatio'
]

print("\n=== FEATURE VARIANCE CHECK ===")
for i, n in enumerate(names):
    var = float(np.var(feats[:, i]))
    flag = " <-- DEAD" if var < 1e-4 else ""
    print(f"dim {i:2d}  {n:22s}  var={var:.8f}{flag}")