"""提覆盖杠杆测试(纯json): 基线绝对残差 conformal vs 归一化(残差/ens_std)局部自适应 conformal.
同 alpha 比较 kept-FP 与 认证覆盖率(安全件中)."""
import json
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
S=Path("/home/lin/xvxinyao/ShadowForge")
d=json.load(open(S/"certification/pino_vs_fem_hard450.json"))
fem=np.array(d['fem_sf'],float);pino=np.array(d['pino_sf'],float)
es=np.array(d.get('ens_std_log10',[np.nan]*len(fem)),float)
m=(fem>0)&(pino>0)&np.isfinite(fem)&np.isfinite(pino)&np.isfinite(es);fem,pino,es=fem[m],pino[m],es[m]
x=np.log10(pino);y=np.log10(fem);TAU=np.log10(1.3);N=len(y)
es=np.clip(es,np.percentile(es,5),None)            # 防除0
print(f"n={N}, 不安全={int((y<TAU).sum())}, ens_std 范围[{es.min():.3f},{es.max():.3f}]")
def run(alpha,seed,normalized):
    rng=np.random.RandomState(seed);idx=rng.permutation(N);h=N//2;ci,ti=idx[:h],idx[h:]
    iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x[ci],y[ci])
    res=iso.predict(x[ci])-y[ci]
    if normalized:
        q=np.quantile(res/es[ci],1-alpha); g=iso.predict(x[ti])-q*es[ti]
    else:
        q=np.quantile(res,1-alpha); g=iso.predict(x[ti])-q
    acc=g>=TAU; truth=y[ti]>=TAU
    return int((acc&~truth).sum()), (acc&truth).sum()/max(truth.sum(),1)
for alpha in (0.05,0.10):
    for norm,name in [(False,"基线(绝对残差)"),(True,"归一化(÷ens_std)")]:
        fps=[];covs=[]
        for sd in range(20):
            fp,cov=run(alpha,sd,norm);fps.append(fp);covs.append(cov)
        print(f"alpha={alpha:.2f} {name:18s}: kept-FP 均值={np.mean(fps):.2f} 最大={max(fps)}  覆盖率={np.mean(covs)*100:.0f}%")
print("DONE")
