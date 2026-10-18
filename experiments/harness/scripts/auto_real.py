import sys, importlib.util, json, os
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT=Path("/data/xvxinyao/xvxinyao/total_project"); sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("v26",ROOT/"scripts/train_v26_shadowart_scale_sf.py")
v26=importlib.util.module_from_spec(spec); spec.loader.exec_module(v26)
from structural.models.load_model import load_surrogate
dev=torch.device("cuda")
INDEX=ROOT/"results/shadowart_fem_hardset_v1/index.json"
BASE=str(ROOT/"checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
AFTER=str(ROOT/"checkpoints/v26_shadowart_scale_sf/best_scale_sf.pt")
def load(ck):
    p,_=load_surrogate(BASE,device=dev); p.enable_enhanced_scale_head()
    c=torch.load(ck,map_location=dev,weights_only=False); p.scale_head_v2.load_state_dict(c["scale_head_v2_state_dict"])
    if c.get("scale_branch_state_dict"):
        for n,s in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(s)
    p.eval(); return p
test=v26.ShadowArtScaleDataset(INDEX,"test"); loader=DataLoader(test,batch_size=8,shuffle=False,num_workers=2,collate_fn=v26.collate)
pino=load(AFTER); pl,tl,ts=[],[],[]
with torch.no_grad():
    for b in loader:
        p=v26.predict_log_vm(pino,b["x"].to(dev),train_branch=True).cpu().numpy()
        pl.extend(p.tolist()); tl.extend(b["log_vm"].numpy().tolist()); ts.extend(b["sf"].numpy().tolist())
pl,tl,ts=np.array(pl),np.array(tl),np.array(ts)
# v26 相对log-sf残差 (与yield约定无关): r = pred_sf_log - true_sf_log = true_log - pred_log
const=np.log10(ts)+tl; psf=const-pl; tsf=np.log10(ts); resid=psf-tsf
ALPHA=0.1; TAU=1.3; tl10=np.log10(TAU)
# conformal 带 (用全部test残差; finite-sample 上分位)
q_safe=np.quantile(resid,1-ALPHA); q_unsafe=np.quantile(-resid,1-ALPHA)
print(f"v26 残差: MAE={np.abs(resid).mean():.3f} bias={resid.mean():+.3f}  conformal q_safe={q_safe:.2f} q_unsafe={q_unsafe:.2f}",flush=True)
# 真实候选 SF 分布 (ShadowForge 500)
real=[]
for f in ['/home/lin/xvxinyao/ShadowForge/results/positive_probe_48.json','/home/lin/xvxinyao/ShadowForge/results/optionB_extra60.json']:
    if os.path.exists(f):
        d=json.load(open(f)); 
        for c in d.get('cases',[]):
            for r in c.get('candidates',[]):
                if r.get('sf_fem') and r['sf_fem']>0: real.append(r['sf_fem'])
real=np.array(real); rlog=np.log10(real)
# bootstrap: 给每个真实候选采样 v26 残差 -> pred -> conformal 判定
rng=np.random.default_rng(0); NB=200; autos=[]; fps=[]
for _ in range(NB):
    rr=rng.choice(resid,size=len(real),replace=True); pred=rlog+rr
    lcb=pred-q_safe; ucb=pred+q_unsafe; safe=lcb>=tl10; unsafe=ucb<tl10
    autos.append(np.mean(safe|unsafe)); fps.append(np.mean(safe&(real<TAU)) if safe.any() else 0)
print(f"=== v26 在真实候选分布(n={len(real)}, 55%明确安全)上的估计 ===",flush=True)
print(f"  自主率={np.mean(autos)*100:.0f}%  假阳={np.mean(fps)*100:.1f}%  (对比: base在难集~31%, base原始假阳13%)")
print("DONE")
