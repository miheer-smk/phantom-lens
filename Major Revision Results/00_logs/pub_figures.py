#!/usr/bin/env python3
"""Publication-quality figures for the revision — renders REAL committed numbers only (no fabrication).
Reads results_clean/*.{json,csv}. Validated colorblind-safe palette. 300 DPI, direct labels, CIs."""
import os,json,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
R="results_clean"; FIG="Major Revision Results/03_figures/publication"; os.makedirs(FIG,exist_ok=True)
BLUE,ORANGE,PURPLE,RED="#2166AC","#E08214","#762A83","#B2182B"; GREY="#52514e"
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':9.5,
 'ytick.labelsize':9.5,'legend.fontsize':9.5,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
 'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'font.family':'DejaVu Sans'})
def grid(ax): ax.grid(True,axis='y',alpha=0.3,lw=0.5); ax.set_axisbelow(True)
def lbl(ax,bars,fmt="{:.3f}",dy=0.004,fs=8.5):
    for b in bars:
        h=b.get_height(); ax.text(b.get_x()+b.get_width()/2,h+dy,fmt.format(h),ha='center',va='bottom',fontsize=fs)
MAN=["Deepfakes","FaceSwap","Face2Face","NeuralTextures"]; mk=lambda s:s.lower()

# ---- Fig 1: per-manipulation AUC 50-D vs 53-D (+CI, +significance) ----
tc=json.load(open(f"{R}/track_c_53D_full.json"))
dl={r['manip']:r for r in pd.read_csv(f"{R}/delong_53vs50.csv").to_dict('records')}
fig,ax=plt.subplots(figsize=(7.2,4.2)); x=np.arange(len(MAN)); w=0.38
a50=[tc[m]['base50'][0] for m in MAN]; a53=[tc[m]['plusG1'][0] for m in MAN]
e50=[[a-tc[m]['base50'][1][0] for a,m in zip(a50,MAN)],[tc[m]['base50'][1][1]-a for a,m in zip(a50,MAN)]]
e53=[[a-tc[m]['plusG1'][1][0] for a,m in zip(a53,MAN)],[tc[m]['plusG1'][1][1]-a for a,m in zip(a53,MAN)]]
b1=ax.bar(x-w/2,a50,w,yerr=e50,capsize=3,color=BLUE,label='50-D (base)',error_kw=dict(lw=1,ecolor=GREY))
b2=ax.bar(x+w/2,a53,w,yerr=e53,capsize=3,color=ORANGE,label='53-D (+G1 mouth)',error_kw=dict(lw=1,ecolor=GREY))
lbl(ax,b1,dy=0.012); lbl(ax,b2,dy=0.012)
for i,m in enumerate(MAN):
    p=dl[m]['p_value'] if m in dl else 1
    s='***' if p<1e-3 else '**' if p<1e-2 else '*' if p<0.05 else 'n.s.'
    yst=max(a50[i]+e50[1][i],a53[i]+e53[1][i])+0.012; ax.text(x[i],yst,s,ha='center',fontsize=11,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(MAN); ax.set_ylabel('AUC (identity-disjoint test)'); ax.set_ylim(0.6,1.06)
ax.set_title('In-distribution detection: 50-D vs 53-D (extended)',pad=14); ax.legend(loc='lower right'); grid(ax)
ax.text(0.01,-0.17,'Error bars: bootstrap 95% CI. Significance: DeLong test (*** p<0.001, ** p<0.01, * p<0.05, n.s. not sig.).',transform=ax.transAxes,fontsize=7.5,color=GREY)
plt.savefig(f"{FIG}/fig1_indist_50v53.png"); plt.close()

# ---- Fig 2: generalization ladder (in-dist -> cross-manip -> cross-dataset) + Xception ----
bl=json.load(open(f"{R}/baseline.json"))['results']; xc=json.load(open(f"{R}/xception_baseline.json"))
fig,ax=plt.subplots(figsize=(7.6,4.2))
cats=['In-dist\n(mean per-manip)','Cross-manip\n(LOMO mean)','Zero-shot\nCeleb-DF','Zero-shot\nWildDeepfake']
prism=[np.mean([bl['regime1_in_distribution'][m]['auc'] for m in bl['regime1_in_distribution']]),
       np.mean([bl['regime2_cross_manip_LOMO'][m]['auc'] for m in bl['regime2_cross_manip_LOMO']]),
       bl['regime3_zero_shot_celebdf']['CelebDF_v2']['auc'],
       json.load(open(f"{R}/zeroshot_wilddeepfake.json"))['auc']]
xcep=[xc['ffpp_test_overall']['auc'],np.nan,xc['celebdf_zeroshot']['auc'],np.nan]
x=np.arange(len(cats)); w=0.38
b1=ax.bar(x-w/2,prism,w,color=BLUE,label='PRISM (physics, CPU)')
b2=ax.bar([x[i]+w/2 for i in range(len(cats)) if not np.isnan(xcep[i])],[v for v in xcep if not np.isnan(v)],w,color=PURPLE,label='Xception (deep, GPU)')
lbl(ax,b1,dy=0.012); lbl(ax,b2,dy=0.012)
ax.axhline(0.5,ls='--',lw=1,color=GREY); ax.text(len(cats)-0.5,0.51,'chance',fontsize=8,color=GREY,ha='right')
ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel('AUC'); ax.set_ylim(0.4,1.02)
ax.set_title('Generalization: in-distribution → cross-manipulation → cross-dataset'); ax.legend(loc='upper right'); grid(ax)
plt.savefig(f"{FIG}/fig2_generalization.png"); plt.close()

# ---- Fig 3: compression c23 vs c40 ----
cp=json.load(open(f"{R}/compression.json"))
pm={r['manipulation']:r for r in cp['per_manip']}
fig,ax=plt.subplots(figsize=(7.2,4.2)); x=np.arange(len(MAN)); w=0.38
c23=[pm[mk(m)]['c23_auc'] for m in MAN]; c40=[pm[mk(m)]['c40_auc'] for m in MAN]
b1=ax.bar(x-w/2,c23,w,color=BLUE,label='c23 (light compression)'); b2=ax.bar(x+w/2,c40,w,color=RED,label='c40 (heavy compression)')
lbl(ax,b1,dy=0.008); lbl(ax,b2,dy=0.008)
for i,m in enumerate(MAN): ax.text(x[i],min(c23[i],c40[i])-0.05,f"Δ{pm[mk(m)]['delta_auc']:+.3f}",ha='center',fontsize=8,color=RED)
ax.set_xticks(x); ax.set_xticklabels(MAN); ax.set_ylabel('AUC'); ax.set_ylim(0.6,1.02)
ax.set_title('Compression robustness (c23 vs c40), per manipulation'); ax.legend(loc='lower left'); grid(ax)
plt.savefig(f"{FIG}/fig3_compression.png"); plt.close()

# ---- Fig 4: pillar-only standalone AUC heatmap (20 pillars x 5 datasets) ----
po=pd.read_csv(f"{R}/pillar_only.csv"); order=list(json.load(open("splits/pillar_map.json")).keys())
piv=po.pivot(index="pillar",columns="dataset",values="pillar_only_auc").reindex(index=order,columns=["Deepfakes","Face2Face","FaceSwap","NeuralTextures","CelebDF"])
fig,ax=plt.subplots(figsize=(6.4,7.6))
im=ax.imshow(piv.values,cmap='YlOrRd',vmin=0.5,vmax=1.0,aspect='auto')
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns,rotation=30,ha='right')
ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index,fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v=piv.values[i,j]; ax.text(j,i,f"{v:.2f}",ha='center',va='center',fontsize=6.5,color='white' if v>0.8 else 'black')
plt.colorbar(im,fraction=0.046,pad=0.04,label='standalone AUC')
ax.set_title('Per-pillar standalone discriminative power'); plt.savefig(f"{FIG}/fig4_pillar_standalone.png"); plt.close()

# ---- Fig 5: SHAP cross-manipulation ranking Spearman ----
ss=json.load(open(f"{R}/shap_stability.json")); sp=ss['per_manip_spearman']
labels=["Deepfakes","Face2Face","FaceSwap","NeuralTextures"]; M=np.eye(4)
for k,v in sp.items():
    a,b=k.split('~'); i,j=labels.index(a),labels.index(b); M[i,j]=v; M[j,i]=v
fig,ax=plt.subplots(figsize=(5.4,4.6)); im=ax.imshow(M,cmap='RdBu_r',vmin=-1,vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels(labels,rotation=30,ha='right'); ax.set_yticks(range(4)); ax.set_yticklabels(labels)
for i in range(4):
    for j in range(4): ax.text(j,i,f"{M[i,j]:.2f}",ha='center',va='center',fontsize=9,color='white' if abs(M[i,j])>0.6 else 'black')
plt.colorbar(im,fraction=0.046,pad=0.04,label="Spearman ρ")
ax.set_title(f"SHAP ranking similarity across manipulations\n(cross-fold ρ={ss['mean_cross_fold_spearman']}, i.e. fold-stable)")
plt.savefig(f"{FIG}/fig5_shap_stability.png"); plt.close()

# ---- Fig 6: calibration real vs fake recall tradeoff ----
cal=json.load(open(f"{R}/calibration.json"))['celebdf']
fig,ax=plt.subplots(figsize=(7.0,4.2)); cfgs=[c['config'] for c in cal]; x=np.arange(len(cfgs)); w=0.38
rr=[c['real_recall'] for c in cal]; fr=[c['fake_recall'] for c in cal]
b1=ax.bar(x-w/2,rr,w,color=BLUE,label='real recall'); b2=ax.bar(x+w/2,fr,w,color=ORANGE,label='fake recall')
lbl(ax,b1,dy=0.01,fs=8); lbl(ax,b2,dy=0.01,fs=8)
ax.set_xticks(x); ax.set_xticklabels([c.replace('_','\n') for c in cfgs],fontsize=8); ax.set_ylabel('recall (Celeb-DF)'); ax.set_ylim(0,1.05)
ax.set_title('Threshold calibration only TRADES real vs fake recall (AUC fixed 0.632)'); ax.legend(loc='upper center',ncol=2); grid(ax)
plt.savefig(f"{FIG}/fig6_calibration.png"); plt.close()

# ---- Fig 7: runtime per-stage ----
rt=json.load(open(f"{R}/runtime.json")); sm=rt['aggregate']['stage_means_s']
fig,ax=plt.subplots(figsize=(6.8,4.0))
stages=['frame_load_s','mediapipe_s','optical_flow_s','rppg_s','other_s']; names=['frame load','MediaPipe','optical flow','rPPG','other temporal']
vals=[sm[s] for s in stages]; bars=ax.barh(names,vals,color=[BLUE,PURPLE,ORANGE,RED,GREY])
for b,v in zip(bars,vals): ax.text(v+0.4,b.get_y()+b.get_height()/2,f"{v:.1f}s",va='center',fontsize=9)
ax.set_xlabel('mean time per video (s), single CPU thread'); ax.invert_yaxis()
ax.set_title(f"Per-stage extraction time  (total {rt['aggregate']['mean_total_extract_s']:.0f}s/video, RTF {rt['aggregate']['mean_rtf']}, {rt['aggregate']['mean_peak_ram_mb']/1000:.1f} GB RAM)")
ax.grid(True,axis='x',alpha=0.3,lw=0.5); plt.savefig(f"{FIG}/fig7_runtime.png"); plt.close()

# ---- Fig 8: PRNU residual method comparison ----
pr=json.load(open(f"{R}/prnu_comparison.json"))['classification']
methods=[r for r in pr if r['residual_method'] in ('median','gaussian','wavelet')]
fig,ax=plt.subplots(figsize=(7.2,4.2)); x=np.arange(len(MAN)); w=0.26; cols=[BLUE,ORANGE,PURPLE]
for k,(mth,c) in enumerate(zip(methods,cols)):
    vals=[mth[f"auc_{mk(m)}"] for m in MAN]; b=ax.bar(x+(k-1)*w,vals,w,color=c,label=mth['residual_method'])
    lbl(ax,b,dy=0.006,fs=7)
ax.set_xticks(x); ax.set_xticklabels(MAN); ax.set_ylabel('AUC'); ax.set_ylim(0.6,1.02)
ax.set_title('PRNU-inspired residual: method robustness (median≈gaussian≈wavelet; BM3D not computed)')
ax.legend(title='residual method',loc='lower left',ncol=3); grid(ax)
plt.savefig(f"{FIG}/fig8_prnu.png"); plt.close()

# ---- Fig 9: rPPG per-condition ----
rp=json.load(open(f"{R}/rppg_comparison.json"))['results']
conds=['overall','c23','c40','low_motion','high_motion','low_illum','high_illum','short_seq','long_seq']
clabels=['overall','c23','c40','low mot','high mot','low illum','high illum','short','long']
fig,ax=plt.subplots(figsize=(8.6,4.2)); x=np.arange(len(conds)); w=0.26; cols=[BLUE,ORANGE,PURPLE]
for k,(row,c) in enumerate(zip(rp,cols)):
    vals=[row.get(cc) if row.get(cc) is not None else np.nan for cc in conds]
    ax.bar(x+(k-1)*w,vals,w,color=c,label=row['method'])
ax.axhline(0.5,ls='--',lw=1,color=GREY); ax.text(len(conds)-0.5,0.505,'chance',fontsize=8,color=GREY,ha='right')
ax.set_xticks(x); ax.set_xticklabels(clabels,rotation=25,ha='right'); ax.set_ylabel('real-vs-fake AUC'); ax.set_ylim(0.4,0.65)
ax.set_title('rPPG discriminative power by condition (weak; POS best & most compression-robust)')
ax.legend(title='rPPG method',loc='upper right',ncol=3); grid(ax)
plt.savefig(f"{FIG}/fig9_rppg.png"); plt.close()

# ---- Fig 10: DeLong significance (full-50 vs top-k) ----
st=json.load(open(f"{R}/statistical_tests.json"))['delong']
sub=[r for r in st if 'full50_vs_top' in r['comparison']]
fig,ax=plt.subplots(figsize=(7.6,4.2))
names=[r['comparison'].replace('full50_vs_','').replace('[','\n[').replace(']','') for r in sub]
diffs=[r['auc_diff'] for r in sub]; sig=[r['p_holm']<0.05 for r in sub]
bars=ax.bar(range(len(sub)),diffs,color=[BLUE if s else GREY for s in sig])
for i,(b,r) in enumerate(zip(bars,sub)):
    st_=('***' if r['p_holm']<1e-3 else '**' if r['p_holm']<1e-2 else '*' if r['p_holm']<0.05 else 'n.s.')
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.003,st_,ha='center',fontsize=9,fontweight='bold')
ax.set_xticks(range(len(sub))); ax.set_xticklabels(names,fontsize=7.5,rotation=0)
ax.set_ylabel('AUC advantage of full-50 model'); ax.set_title('Full 50-feature model vs reduced sets (DeLong, Holm-corrected)')
ax.legend(handles=[Patch(color=BLUE,label='significant (p_holm<0.05)'),Patch(color=GREY,label='n.s.')],loc='upper right'); grid(ax)
plt.savefig(f"{FIG}/fig10_delong_topk.png"); plt.close()

print("saved 10 publication figures to",FIG)
for f in sorted(os.listdir(FIG)): print("  ",f)
