import json, numpy as np
from sklearn.isotonic import IsotonicRegression
rng=np.random.RandomState(0)
rows=json.load(open("/tmp/boundary_pairs.json"))
pino=np.array([r["pino_sf"] for r in rows],float); fem=np.array([r["fem_sf"] for r in rows],float)
ok=np.isfinite(pino)&np.isfinite(fem)&(pino>0)&(fem>0); pino,fem=pino[ok],fem[ok]
# FDM 有效强度重标定: SF' = SF*(sigma_eff/50). 选 sigma_eff 使边界落入密集区
for SIG in (12.0,):
    k=SIG/50.0
    mu=np.log10(pino*k); Y=np.log10(fem*k); N=len(Y)
    TAU=1.3; LT=np.log10(TAU); ALPHA=0.05
    nb=((fem*k>0.8)&(fem*k<2.0)).sum()
    print(f"sigma_eff={SIG}MPa  n={N}  FEM'<1.3:{(fem*k<TAU).mean()*100:.0f}%  near-bdry(0.8-2.0):{nb}",flush=True)
    # 优化器诱导分布: 越靠近边界(mu≈LT)权重越高
    shift_w=np.exp(-((mu-LT)/0.15)**2)
    def acc_fit(ci,wts=None):
        idx=rng.permutation(len(ci)); h=len(idx)//2; iso_i,cf_i=ci[idx[:h]],ci[idx[h:]]
        g=IsotonicRegression(out_of_bounds="clip").fit(mu[iso_i],Y[iso_i])
        res=np.maximum(0,g.predict(mu[cf_i])-Y[cf_i])
        if wts is None:
            q=np.sort(res)[min(int(np.ceil((1-ALPHA)*(len(cf_i)+1)))-1,len(res)-1)]
        else:
            w=wts[cf_i]; o=np.argsort(res); rs,ws=res[o],w[o]; cw=np.cumsum(ws)/ws.sum()
            q=rs[np.searchsorted(cw,1-ALPHA)] if (cw>=1-ALPHA).any() else rs[-1]
        return g,q
    def fp(g,q,ti):
        acc=(g.predict(mu[ti])-q)>=LT; uns=Y[ti]<LT
        return (acc&uns).sum()/max(acc.sum(),1)
    rn,rns,rss=[],[],[]
    for t in range(40):
        pm=rng.permutation(N); cal,te=pm[:N//2],pm[N//2:]
        gn,qn=acc_fit(cal)
        rn.append(fp(gn,qn,te))                                  # 朴素@iid
        wt=shift_w[te]/shift_w[te].sum(); samp=rng.choice(te,len(te),p=wt)
        rns.append(fp(gn,qn,samp))                               # 朴素@漂移
        gs,qs=acc_fit(cal,wts=shift_w)
        rss.append(fp(gs,qs,samp))                               # 漂移感知@漂移
    print(f"  朴素ACCS @ iid分布:        {np.mean(rn)*100:.1f}% ± {np.std(rn)*100:.1f}%")
    print(f"  朴素ACCS @ 优化诱导漂移:   {np.mean(rns)*100:.1f}% ± {np.std(rns)*100:.1f}%  <-应飙升")
    print(f"  漂移感知ACCS @ 漂移:       {np.mean(rss)*100:.1f}% ± {np.std(rss)*100:.1f}%  <-应回落≤{ALPHA*100:.0f}%",flush=True)
