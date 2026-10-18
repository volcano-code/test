"""新定义验证: 单边 conformal 直接作用裸分数(无isotonic)=canonical ACS.
验证有限样本保证: 经验 边际FP率 = Pr[certify & unsafe] <= alpha, 跨 alpha 扫描. + 覆盖."""
import json
from pathlib import Path
import numpy as np
S=Path("/home/lin/xvxinyao/ShadowForge")
d=json.load(open(S/"certification/pino_vs_fem_hard450.json"))
fem=np.array(d['fem_sf'],float);pino=np.array(d['pino_sf'],float)
m=(fem>0)&(pino>0)&np.isfinite(fem)&np.isfinite(pino);fem,pino=fem[m],pino[m]
x=np.log10(pino);y=np.log10(fem);TAU=np.log10(1.3);N=len(y)
def run(alpha,seed):
    rng=np.random.RandomState(seed);idx=rng.permutation(N);h=N//2;ci,ti=idx[:h],idx[h:]
    q=np.quantile(x[ci]-y[ci],1-alpha)              # 单边分位数, 裸分数(无isotonic)
    acc=(x[ti]-q)>=TAU; truth=y[ti]>=TAU
    fp_rate=(acc&~truth).sum()/len(ti)              # 边际 FP 率 = Pr[certify & unsafe]
    cov=(acc&truth).sum()/max(truth.sum(),1)
    return fp_rate,cov
print(f"单边 conformal on 裸分数 (无isotonic), n={N}, 不安全占比={np.mean(y<TAU):.2f}")
print(f"{'alpha':>6}{'经验FP率(均值)':>14}{'FP率<=alpha?':>14}{'覆盖率':>9}")
for alpha in (0.02,0.05,0.10,0.15,0.20):
    fps=[];covs=[]
    for s in range(50): fp,c=run(alpha,s);fps.append(fp);covs.append(c)
    mf=np.mean(fps);ok="是" if mf<=alpha+1e-9 else "否"
    print(f"{alpha:>6.2f}{mf:>14.4f}{ok:>12}{100*np.mean(covs):>8.0f}%")
print("DONE")
