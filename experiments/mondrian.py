import sys,json,numpy as np,torch,importlib.util
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge")
from sklearn.isotonic import IsotonicRegression
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from structural.analysis.load_generator import LoadGenerator
dev=torch.device('cuda');TAU=1.3;LOGTHR=np.log10(TAU);CP="/home/lin/xvxinyao/ShadowForge/checkpoints";lg=LoadGenerator()
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
c=torch.load("/home/lin/xvxinyao/ShadowForge/data/v40_finetuned_both.pt",map_location=dev,weights_only=False)
for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
h=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));h.load_state_dict(c["attn_scale_head_state_dict"]);p.eval()
def raw_of(d,f):
    x=build_x(d.astype(np.float32),f.astype(np.float32)).unsqueeze(0).to(dev)
    b,cc,nx,ny,nz=x.shape;g=p._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1)
    with torch.no_grad(): return float(p._scale_forward(torch.cat([x,g],1)).squeeze().cpu())
HARD=__import__('pathlib').Path(ROOT)/"results/shadowart_fem_hardset_v1"
dd=json.load(open("/home/lin/xvxinyao/ShadowForge/certification/pino_vs_fem_hard450.json"));paths=dd['paths'];fem=np.array(dd['fem_sf'])
ix=np.arange(len(paths));np.random.seed(0);np.random.shuffle(ix)
af=np.load("/home/lin/xvxinyao/ShadowForge/data/aug.npz");ax=np.load("/home/lin/xvxinyao/ShadowForge/data/aug_fixed.npz")
pf=np.random.RandomState(2).permutation(len(af['dens']));px=np.random.RandomState(3).permutation(len(ax['dens']))
ftr,fte=pf[:440],pf[440:];xtr,xte=px[:440],px[440:]
def collect(items):
    R=[];Y=[]
    for kind,a,j in items:
        if kind=='hs':
            d=np.load(HARD/paths[j],allow_pickle=True);R.append(raw_of(d['density'],d['force']));Y.append(np.log10(max(fem[j],1e-6)))
        else:
            v=a['dens'][j].astype(np.float32);f=lg.standing_load(v)["force"]*a['mult'][j];R.append(raw_of(v,f));Y.append(np.log10(max(a['sf'][j],1e-6)))
    return np.array(R),np.array(Y)
cal_items=[('hs',None,i) for i in ix[:120]]+[('a',af,j) for j in ftr[:150]]+[('a',ax,j) for j in xtr[:150]]
te_items=[('a',af,j) for j in fte]+[('a',ax,j) for j in xte]
print("算校准/测试raw...",flush=True)
cr,cy=collect(cal_items);tr,ty=collect(te_items)
o=np.argsort(cr);crs,cys=cr[o],cy[o];hh=len(crs)//2
iso=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(crs[:hh],cys[:hh])
# 用后半校准做conformal
cpred=iso.predict(crs[hh:]);cresid=np.maximum(cpred-cys[hh:],0)
tpred=iso.predict(tr);truth=ty>=LOGTHR;ns=int(truth.sum())
def cov_fp(acc): return int((acc&truth).sum())/max(ns,1), int((acc&~truth).sum())/max(len(ty)-ns,1)
for a in (0.05,0.10,0.20):
    qg=np.quantile(cresid,1-a);accg=tpred-qg>=LOGTHR
    # Mondrian: 按预测SF分3档, 每档自己的q
    bins=np.quantile(cpred,[0.33,0.66]);cbin=np.digitize(cpred,bins);tbin=np.digitize(tpred,bins)
    accm=np.zeros(len(tr),bool)
    for b in range(3):
        rb=cresid[cbin==b]; qb=np.quantile(rb,1-a) if len(rb)>5 else qg
        m=tbin==b;accm[m]=tpred[m]-qb>=LOGTHR
    cg,fg=cov_fp(accg);cm,fm=cov_fp(accm)
    print(f"α={a}: 全局 cov={cg:.0%} FP={fg:.1%} | Mondrian cov={cm:.0%} FP={fm:.1%}",flush=True)
print(f"(测试 n={len(ty)} 安全{ns} corr={np.corrcoef(tr,ty)[0,1]:.3f})")
print("DONE")
