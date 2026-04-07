import pickle
import numpy as np

with open('data/precomputed_features.pkl', 'rb') as f:
    data = pickle.load(f)

features = np.array(data['features'])
labels   = np.array(data['labels'])
sources  = np.array(data['dataset_sources'])

ffpp_fake  = features[(sources=='ffpp_official') & (labels==1)]
ffpp_real  = features[(sources=='ffpp_official') & (labels==0)]
celeb_real = features[(sources=='celebvhq')      & (labels==0)]

print('='*65)
print('FEATURE DISTRIBUTION DIAGNOSIS')
print('='*65)
print(f'Samples — ffpp_fake: {len(ffpp_fake)} | ffpp_real: {len(ffpp_real)} | celeb_real: {len(celeb_real)}')
print()

dim_names = [
    'P1_vmr','P1_res_std','P1_hf_ratio',
    'P2_prnu_energy','P2_face_peri',
    'P3_rg_corr','P3_bg_corr',
    'P4_face_bg','P4_specular','P4_shadow',
    'P5_drift_mean','P5_drift_std',
    'P6_benford','P6_block_art','P6_dbl_comp',
    'P7_res_mean','P7_res_var',
    'P8_blur_mag','P8_blur_dir',
    'P9_flow_mag','P9_flow_bnd',
    'P10_rg_shift','P10_bg_shift','P10_edge_ctr'
]

print(f'{"Dim":<18} {"FF++_fake_mean":>15} {"FF++_real_mean":>15} {"celeb_real_mean":>16} {"fake-real_gap":>14}')
print('-'*80)
for i, name in enumerate(dim_names):
    fm  = ffpp_fake[:,i].mean()
    rm  = ffpp_real[:,i].mean()
    cm  = celeb_real[:,i].mean()
    gap = fm - rm
    print(f'{name:<18} {fm:>15.4f} {rm:>15.4f} {cm:>16.4f} {gap:>14.4f}')

print()
print('='*65)
print('STANDARD DEVIATIONS')
print('='*65)
print(f'{"Dim":<18} {"FF++_fake_std":>14} {"FF++_real_std":>14} {"celeb_real_std":>15}')
print('-'*65)
for i, name in enumerate(dim_names):
    fs = ffpp_fake[:,i].std()
    rs = ffpp_real[:,i].std()
    cs = celeb_real[:,i].std()
    print(f'{name:<18} {fs:>14.4f} {rs:>14.4f} {cs:>15.4f}')

print()
print('='*65)
print('DEAD FEATURE CHECK (std near zero = dead)')
print('='*65)
for i, name in enumerate(dim_names):
    fs = ffpp_fake[:,i].std()
    rs = ffpp_real[:,i].std()
    if fs < 0.001 or rs < 0.001:
        print(f'  DEAD: {name} — fake_std={fs:.6f} real_std={rs:.6f}')