import sys, importlib.util
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
SCALE_BEFORE=str(ROOT/"checkpoints/scale_head_v2_weighted/best_scale_head_v2_weighted.pt")
SCALE_AFTER=str(ROOT/"checkpoints/v26_shadowart_scale_sf/best_scale_sf.pt")
def load(ck_path):
    pino,_=load_surrogate(BASE,device=dev); pino.enable_enhanced_scale_head()
    ck=torch.load(ck_path,map_location=dev,weights_only=False)
    pino.scale_head_v2.load_state_dict(ck["scale_head_v2_state_dict"])
    if ck.get("scale_branch_state_dict"):
        for n,s in ck["scale_branch_state_dict"].items(): getattr(pino,n).load_state_dict(s)
    pino.eval(); return pino
test=v26.ShadowArtScaleDataset(INDEX,"test")
loader=DataLoader(test,batch_size=8,shuffle=False,num_workers=2,collate_fn=v26.collate)
def predict(pino):
    pl,tl,ts=[],[],[]
    with torch.no_grad():
        for b in loader:
            p=v26.predict_log_vm(pino,b["x"].to(dev),train_branch=True).detach().cpu().numpy()
            pl.extend(p.tolist()); tl.extend(b["log_vm"].numpy().tolist()); ts.extend(b["sf"].numpy().tolist())
    return np.array(pl),np.array(tl),np.array(ts)
TAU=1.3; tl10=np.log10(TAU); ALPHA=0.1
def autonomy(pl,tlg,ts):
    const=np.log10(ts)+tlg; psf=const-pl; tsf=np.log10(ts)
    n=len(psf); idx=np.random.default_rng(0).permutation(n); ca,ev=idx[:n//2],idx[n//2:]
    qs=np.quantile(psf[ca]-tsf[ca],1-ALPHA); qu=np.quantile(tsf[ca]-psf[ca],1-ALPHA)
    lcb=psf[ev]-qs; ucb=psf[ev]+qu; safe=lcb>=tl10; unsafe=ucb<tl10
    auto=np.mean(safe|unsafe); fp=np.mean(safe&(ts[ev]<TAU)) if safe.any() else 0.0
    return auto,fp,np.mean(np.abs(psf-tsf))
for tag,ck in [("before(base+原scale头)",SCALE_BEFORE),("after(v26微调头)",SCALE_AFTER)]:
    pino=load(ck); pl,tlg,ts=predict(pino); a,fp,mae=autonomy(pl,tlg,ts)
    print(f"{tag}: 自主率={a*100:.0f}%  假阳={fp*100:.1f}%  log_sf_MAE={mae:.3f}  (n_test={len(pl)})",flush=True)
    del pino; torch.cuda.empty_cache()
print("DONE")
