"""ACCS 组件消融(纯json): raw / isotonic-only / conformal-only / both. alpha=0.10, 20 split.
显示: conformal 控 FP, isotonic 保覆盖(校正尺度偏置)."""
import json
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
S=Path("/home/lin/xvxinyao/ShadowForge")
d=json.load(open(S/"certification/pino_vs_fem_hard450.json"))
fem=np.array(d['fem_sf'],float);pino=np.array(d['pino_sf'],float)
m=(fem>0)&(pino>0)&np.isfinite(fem)&np.isfinite(pino);fem,pino=fem[m],pino[m]
x=np.log10(pino);y=np.log10(fem);TAU=np.log10(1.3);N=len(y);A=0.10
def metrics(acc,truth):
    return int((acc&~truth).sum()), (acc&truth).sum()/max(truth.sum(),1)
def run(variant,seed):
    rng=np.random.RandomState(seed);idx=rng.permutation(N);h=N//2;ci,ti=idx[:h],idx[h:]
    truth=y[ti]>=TAU
    if variant=="raw":                       # 无标定无conformal: 直接 pino>=tau
        acc=x[ti]>=TAU
    elif variant=="iso":                     # 仅isotonic点估计
        iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x[ci],y[ci])
        acc=iso.predict(x[ti])>=TAU
    elif variant=="conf":                    # 仅conformal(无isotonic): raw 残差边界
        q=np.quantile(x[ci]-y[ci],1-A); acc=(x[ti]-q)>=TAU
    else:                                    # both = ACCS
        iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x[ci],y[ci])
        q=np.quantile(iso.predict(x[ci])-y[ci],1-A); acc=(iso.predict(x[ti])-q)>=TAU
    return metrics(acc,truth)
print(f"n={N}, 不安全={int((y<TAU).sum())}, alpha={A}")
print(f"{'变体':>26}{'kept-FP均值':>12}{'最大':>6}{'覆盖率':>8}")
for v,name in [("raw","raw(无标定无conformal)"),("iso","仅isotonic"),("conf","仅conformal(无iso)"),("both","both=ACCS")]:
    fps=[];cov=[]
    for s in range(20): fp,c=run(v,s);fps.append(fp);cov.append(c)
    print(f"{name:>26}{np.mean(fps):>12.2f}{max(fps):>6}{100*np.mean(cov):>7.0f}%")
print("DONE")
