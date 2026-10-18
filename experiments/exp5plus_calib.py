import sys,json,numpy as np,torch,importlib.util
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge")
from sklearn.isotonic import IsotonicRegression
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
dev=torch.device('cuda');TAU=1.3;LOGTHR=np.log10(TAU);CP="/home/lin/xvxinyao/ShadowForge/checkpoints"
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
c=torch.load(f"{CP}/v40_geoattn/best.pt",map_location=dev,weights_only=False)
for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
hd=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));hd.load_state_dict(c["attn_scale_head_state_dict"]);p.eval()
def praw(d,f):
    x=build_x(d.astype(np.float32),f.astype(np.float32)).unsqueeze(0).to(dev)
    b,cc,nx,ny,nz=x.shape;g=p._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1)
    with torch.no_grad(): return float(p._scale_forward(torch.cat([x,g],1)).squeeze().cpu())
# hardset 校准
HARD=__import__('pathlib').Path(ROOT)/"results/shadowart_fem_hardset_v1"
dd=json.load(open("/home/lin/xvxinyao/ShadowForge/certification/pino_vs_fem_hard450.json"));paths=dd['paths'];fem=np.array(dd['fem_sf'])
idx=np.arange(len(paths));np.random.seed(0);np.random.shuffle(idx);ci=idx[:160]
hs_raw=[];hs_Y=[]
for i in ci:
    d=np.load(HARD/paths[i],allow_pickle=True);hs_raw.append(praw(d['density'],d['force']));hs_Y.append(np.log10(max(fem[i],1e-6)))
hs_raw=np.array(hs_raw);hs_Y=np.array(hs_Y)
# 零空间(已存)
ns=np.load("/tmp/nullspace_femlabels.npz");ns_raw=ns['raw'];ns_sf=ns['sf'];ns_Y=np.log10(np.maximum(ns_sf,1e-6))
np.random.seed(2);perm=np.random.permutation(len(ns_raw));half=len(perm)//2
ns_tr,ns_te=perm[:half],perm[half:]
def accs_fit(R,Y):
    o=np.argsort(R);R,Y=R[o],Y[o];h=len(R)//2
    iso=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(R[:h],Y[:h])
    res=np.maximum(iso.predict(R[h:])-Y[h:],0);qm=np.quantile(res,0.95);return iso,qm
def eval_te(iso,qm,R,Y):
    acc=iso.predict(R)-qm>=LOGTHR;truth=Y>=LOGTHR
    fp=int((acc&~truth).sum());tp=int((acc&truth).sum());nsafe=int(truth.sum())
    return fp,tp,nsafe
print(f"零空间测试集: {len(ns_te)}样本, 安全{int((ns_Y[ns_te]>=LOGTHR).sum())}",flush=True)
# 基线: 仅hardset校准
iso0,qm0=accs_fit(hs_raw,hs_Y);fp0,tp0,ns0=eval_te(iso0,qm0,ns_raw[ns_te],ns_Y[ns_te])
# 增广: hardset+零空间训练半
iso1,qm1=accs_fit(np.concatenate([hs_raw,ns_raw[ns_tr]]),np.concatenate([hs_Y,ns_Y[ns_tr]]))
fp1,tp1,ns1=eval_te(iso1,qm1,ns_raw[ns_te],ns_Y[ns_te])
print(f"\n基线(仅hardset校准):   误放行FP={fp0} 安全认证率={tp0}/{ns0}")
print(f"增广(+零空间校准):     误放行FP={fp1} 安全认证率={tp1}/{ns1}")
print("DONE")
