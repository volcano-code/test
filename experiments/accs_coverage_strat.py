"""按真实安全裕度分层报覆盖(非对抗代表性数). 基线ACCS, alpha=0.10, 20 split."""
import json
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
S=Path("/home/lin/xvxinyao/ShadowForge")
d=json.load(open(S/"certification/pino_vs_fem_hard450.json"))
fem=np.array(d['fem_sf'],float);pino=np.array(d['pino_sf'],float)
m=(fem>0)&(pino>0)&np.isfinite(fem)&np.isfinite(pino);fem,pino=fem[m],pino[m]
x=np.log10(pino);y=np.log10(fem);TAU=np.log10(1.3);N=len(y);ALPHA=0.10
# 安全裕度分层(fem_sf 相对 1.3)
bins=[(1.3,1.5),(1.5,2.0),(2.0,3.0),(3.0,1e9)]
acc_by_bin={b:[] for b in bins}; fp_all=[]
for sd in range(20):
    rng=np.random.RandomState(sd);idx=rng.permutation(N);h=N//2;ci,ti=idx[:h],idx[h:]
    iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x[ci],y[ci])
    q=np.quantile(iso.predict(x[ci])-y[ci],1-ALPHA)
    g=iso.predict(x[ti])-q; acc=g>=TAU; truth=y[ti]>=TAU
    fp_all.append(int((acc&~truth).sum()))
    fsf=fem[ti]
    for lo,hi in bins:
        sel=(fsf>=lo)&(fsf<hi)
        if sel.sum()>0: acc_by_bin[(lo,hi)].append(acc[sel].mean())
print(f"n={N}, alpha={ALPHA}, kept-FP 均值={np.mean(fp_all):.2f} 最大={max(fp_all)}")
print("=== 认证覆盖率 按真实安全裕度分层 ===")
for (lo,hi) in bins:
    cs=acc_by_bin[(lo,hi)];cnt=int(((fem>=lo)&(fem<hi)).sum())
    label=f"SF∈[{lo},{hi})" if hi<1e9 else f"SF≥{lo}"
    print(f"  {label:14s} (n={cnt:3d}): 覆盖率={100*np.mean(cs):.0f}%")
# 综合: 明显安全 (SF>=1.5) 的覆盖
for thr in (1.5,2.0):
    cov=[];
    for sd in range(20):
        rng=np.random.RandomState(sd);idx=rng.permutation(N);h=N//2;ci,ti=idx[:h],idx[h:]
        iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x[ci],y[ci])
        q=np.quantile(iso.predict(x[ci])-y[ci],1-ALPHA)
        acc=(iso.predict(x[ti])-q)>=TAU; sel=fem[ti]>=thr
        if sel.sum()>0: cov.append(acc[sel].mean())
    print(f"  明显安全 SF≥{thr}: 覆盖率={100*np.mean(cov):.0f}%")
print("DONE")
