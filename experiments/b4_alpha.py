import sys,json,numpy as np,torch,importlib.util
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge")
from sklearn.isotonic import IsotonicRegression
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from structural.analysis.load_generator import LoadGenerator
dev=torch.device('cuda');TAU=1.3;LOGTHR=np.log10(TAU);CP="/home/lin/xvxinyao/ShadowForge/checkpoints";lg=LoadGenerator()
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
c=torch.load("/home/lin/xvxinyao/ShadowForge/data/v40_finetuned.pt",map_location=dev,weights_only=False)
for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
h=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));h.load_state_dict(c["attn_scale_head_state_dict"]);p.eval()
def raw_of(d,f):
    x=build_x(d.astype(np.float32),f.astype(np.float32)).unsqueeze(0).to(dev)
    b,cc,nx,ny,nz=x.shape;g=p._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1)
    with torch.no_grad(): return float(p._scale_forward(torch.cat([x,g],1)).squeeze().cpu())
aug=np.load("/home/lin/xvxinyao/ShadowForge/data/aug.npz");te=np.load("/home/lin/xvxinyao/ShadowForge/data/test_idx.npy")
A_d,A_m,A_sf=aug['dens'],aug['mult'],aug['sf']
HARD=__import__('pathlib').Path(ROOT)/"results/shadowart_fem_hardset_v1"
dd=json.load(open("/home/lin/xvxinyao/ShadowForge/certification/pino_vs_fem_hard450.json"));paths=dd['paths'];fem=np.array(dd['fem_sf'])
ix=np.arange(len(paths));np.random.seed(0);np.random.shuffle(ix)
cr=[];cy=[]
for i in ix[:120]:
    d=np.load(HARD/paths[i],allow_pickle=True);cr.append(raw_of(d['density'],d['force']));cy.append(np.log10(max(fem[i],1e-6)))
allidx=list(set(range(len(A_d)))-set(te.tolist()))[:200]
for j in allidx:
    v=A_d[j].astype(np.float32);f=lg.standing_load(v)["force"]*A_m[j];cr.append(raw_of(v,f));cy.append(np.log10(max(A_sf[j],1e-6)))
cr=np.array(cr);cy=np.array(cy);o=np.argsort(cr);crs,cys=cr[o],cy[o];hh=len(crs)//2
iso=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(crs[:hh],cys[:hh])
resid=np.maximum(iso.predict(crs[hh:])-cys[hh:],0)
# 测试集
tr=[];ty=[]
for j in te:
    v=A_d[j].astype(np.float32);f=lg.standing_load(v)["force"]*A_m[j];tr.append(raw_of(v,f));ty.append(np.log10(max(A_sf[j],1e-6)))
tr=np.array(tr);ty=np.array(ty);truth=ty>=LOGTHR;nsafe=int(truth.sum());nunsafe=len(ty)-nsafe
print(f"留出零空间: {len(te)}样本 安全{nsafe}/不安全{nunsafe} (微调PINO corr={np.corrcoef(tr,ty)[0,1]:.3f})")
print(f"{'α':>6}{'q⁻':>7}{'认证率(coverage)':>16}{'误放行FP率':>11}")
for a in (0.01,0.05,0.10,0.20,0.30):
    qm=np.quantile(resid,1-a);acc=iso.predict(tr)-qm>=LOGTHR
    cov=int((acc&truth).sum())/max(nsafe,1);fpr=int((acc&~truth).sum())/max(nunsafe,1)
    print(f"{a:>6.2f}{qm:>7.3f}{int((acc&truth).sum())}/{nsafe} ={cov:>6.1%}{fpr:>10.1%}",flush=True)
print("DONE")
