"""汇总版自包含重验 ACCS: 纯用 certification/pino_vs_fem_hard450.json (无需3.7G raw).
isotonic 标定 + 单边 conformal q + 决策 certify iff g(raw)-q>=log(tau). 统计 kept-FP / 认证率."""
import sys,json
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
S=Path("/home/lin/xvxinyao/ShadowForge")
d=json.load(open(S/"certification/pino_vs_fem_hard450.json"))
fem=np.array(d['fem_sf'],float);pino=np.array(d['pino_sf'],float)
m=(fem>0)&(pino>0)&np.isfinite(fem)&np.isfinite(pino);fem,pino=fem[m],pino[m]
x=np.log10(pino);y=np.log10(fem);TAU=np.log10(1.3);N=len(y)
print(f"Hard450 可用 n={N}, 真实不安全(SF<1.3): {int((y<TAU).sum())}")
def run(alpha,seed):
    rng=np.random.RandomState(seed);idx=rng.permutation(N);h=N//2
    ci,ti=idx[:h],idx[h:]
    iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x[ci],y[ci])
    res=iso.predict(x[ci])-y[ci]                       # 过预测残差
    q=np.quantile(res,1-alpha)                         # 单边上分位
    g=iso.predict(x[ti])-q                             # 保守校准预测
    acc=g>=TAU; truth=y[ti]>=TAU
    fp=int((acc&~truth).sum()); kept=int(acc.sum())
    cov=(acc&truth).sum()/max(truth.sum(),1)
    return fp,kept,float(cov),int(truth.sum())
for alpha in (0.05,0.10,0.20):
    fps=[];covs=[];kept=[]
    for s in range(20):
        fp,k,cov,nsafe=run(alpha,s);fps.append(fp);covs.append(cov);kept.append(k)
    print(f"alpha={alpha:.2f}: 20随机split  kept-FP 均值={np.mean(fps):.2f} 最大={max(fps)}  "
          f"认证覆盖率(安全中被认证)={np.mean(covs)*100:.0f}%")
print("DONE")
