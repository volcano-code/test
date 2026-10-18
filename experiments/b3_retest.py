import sys,json,numpy as np,torch,importlib.util
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge")
from sklearn.isotonic import IsotonicRegression
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from structural.analysis.load_generator import LoadGenerator
dev=torch.device('cuda');TAU=1.3;LOGTHR=np.log10(TAU);CP="/home/lin/xvxinyao/ShadowForge/checkpoints";lg=LoadGenerator()
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
SB=["scale_lifting","scale_down1","scale_enc2","scale_down2","scale_bottleneck"]
def load_model(ck):
    p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
    c=torch.load(ck,map_location=dev,weights_only=False)
    for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
    h=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));h.load_state_dict(c["attn_scale_head_state_dict"]);p.eval()
    return p
def raw_of(p,d,f):
    x=build_x(d.astype(np.float32),f.astype(np.float32)).unsqueeze(0).to(dev)
    b,cc,nx,ny,nz=x.shape;g=p._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1)
    with torch.no_grad(): return float(p._scale_forward(torch.cat([x,g],1)).squeeze().cpu())
aug=np.load("/home/lin/xvxinyao/ShadowForge/data/aug.npz");te=np.load("/home/lin/xvxinyao/ShadowForge/data/test_idx.npy")
A_d,A_m,A_vm,A_sf=aug['dens'],aug['mult'],aug['vm'],aug['sf']
def eval_model(ck,calib_aug,tag):
    p=load_model(ck)
    # 留出测试 raw + 真值
    te_raw=[];te_Y=[]
    for j in te:
        v=A_d[j].astype(np.float32);f=lg.standing_load(v)["force"]*A_m[j]
        te_raw.append(raw_of(p,v,f));te_Y.append(np.log10(max(A_sf[j],1e-6)))
    te_raw=np.array(te_raw);te_Y=np.array(te_Y)
    corr=np.corrcoef(te_raw,te_Y)[0,1]
    # 校准: hardset + (可选)零空间训练集
    HARD=__import__('pathlib').Path(ROOT)/"results/shadowart_fem_hardset_v1"
    dd=json.load(open("/home/lin/xvxinyao/ShadowForge/certification/pino_vs_fem_hard450.json"));paths=dd['paths'];fem=np.array(dd['fem_sf'])
    ix=np.arange(len(paths));np.random.seed(0);np.random.shuffle(ix)
    cr=[];cy=[]
    for i in ix[:120]:
        d=np.load(HARD/paths[i],allow_pickle=True);cr.append(raw_of(p,d['density'],d['force']));cy.append(np.log10(max(fem[i],1e-6)))
    if calib_aug:
        tr=np.load("/home/lin/xvxinyao/ShadowForge/data/test_idx.npy");allidx=set(range(len(A_d)))-set(te.tolist())
        for j in list(allidx)[:200]:
            v=A_d[j].astype(np.float32);f=lg.standing_load(v)["force"]*A_m[j];cr.append(raw_of(p,v,f));cy.append(np.log10(max(A_sf[j],1e-6)))
    cr=np.array(cr);cy=np.array(cy);o=np.argsort(cr);cr,cy=cr[o],cy[o];h=len(cr)//2
    iso=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(cr[:h],cy[:h])
    res=np.maximum(iso.predict(cr[h:])-cy[h:],0);qm=np.quantile(res,0.95)
    acc=iso.predict(te_raw)-qm>=LOGTHR;truth=te_Y>=LOGTHR
    fp=int((acc&~truth).sum());tp=int((acc&truth).sum());nsafe=int(truth.sum())
    print(f"[{tag}] 留出零空间相关={corr:.3f} | FP={fp}(真不安全被认证) 认证率={tp}/{nsafe}安全 总{len(te)}",flush=True)
print("=== 微调前(原v40a, 仅hardset校准) ===")
eval_model(f"{CP}/v40_geoattn/best.pt",False,"baseline")
print("=== 微调后(+零空间, hardset+零空间校准) ===")
eval_model("/home/lin/xvxinyao/ShadowForge/data/v40_finetuned.pt",True,"finetuned")
print("DONE")
