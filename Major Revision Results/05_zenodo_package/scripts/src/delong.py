#!/usr/bin/env python3
"""Fast DeLong test for paired AUC comparison (Sun & Xu 2014 algorithm).
delong_roc_test(y, pred_a, pred_b) -> (auc_a, auc_b, z, p_two_sided) for H0: AUC_a == AUC_b.
Paired: both prediction vectors on the SAME samples y (0/1)."""
import numpy as np
from scipy.stats import norm

def _midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N)
    i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2

def _fast_delong(preds_sorted, m):
    # preds_sorted: k x N, positives first (m positives)
    n=preds_sorted.shape[1]-m; k=preds_sorted.shape[0]
    pos=preds_sorted[:,:m]; neg=preds_sorted[:,m:]
    tx=np.empty([k,m]); ty=np.empty([k,n]); tz=np.empty([k,m+n])
    for r in range(k):
        tx[r]=_midrank(pos[r]); ty[r]=_midrank(neg[r]); tz[r]=_midrank(preds_sorted[r])
    aucs=tz[:,:m].sum(1)/m/n-(m+1.0)/2.0/n
    v01=(tz[:,:m]-tx)/n; v10=1.0-(tz[:,m:]-ty)/m
    sx=np.cov(v01); sy=np.cov(v10)
    cov=sx/m+sy/n
    return aucs, cov

def delong_roc_test(y, pa, pb):
    y=np.asarray(y).astype(int); pa=np.asarray(pa,float); pb=np.asarray(pb,float)
    order=np.argsort(-y)  # positives (1) first
    m=int(y.sum())
    preds=np.vstack((pa,pb))[:,order]
    aucs,cov=_fast_delong(preds,m)
    var=cov[0,0]+cov[1,1]-2*cov[0,1]
    if var<=0: var=1e-12
    z=(aucs[0]-aucs[1])/np.sqrt(var)
    p=2*(1-norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)

def holm(pvals):
    """Holm-Bonferroni corrected p-values (list in original order)."""
    m=len(pvals); order=np.argsort(pvals); out=[0.0]*m; prev=0.0
    for rank,idx in enumerate(order):
        adj=min(1.0,(m-rank)*pvals[idx]); adj=max(adj,prev); prev=adj; out[idx]=adj
    return out
