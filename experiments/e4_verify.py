import sys,numpy as np,torch,importlib.util
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+"/scripts")
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from structural.analysis.load_generator import LoadGenerator
dev=torch.device('cuda');lg=LoadGenerator();CP=ROOT+"/checkpoints"
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
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
def corr_on(aug_path,ck,tag):
    aug=np.load(aug_path);te=np.load(ROOT+"/data/test_idx.npy")
    A_d,A_m,A_sf=aug['dens'],aug['mult'],aug['sf']
    p=load_model(ck);raw=[];Y=[]
    for j in te:
        v=A_d[j].astype(np.float32);f=lg.standing_load(v)["force"]*A_m[j]
        raw.append(raw_of(p,v,f));Y.append(np.log10(max(A_sf[j],1e-6)))
    raw=np.array(raw);Y=np.array(Y);c=np.corrcoef(raw,Y)[0,1]
    print(f"[{tag}] 留出零空间 raw↔真值SF 相关 = {c:.3f}  (n={len(te)})",flush=True)
    return c
print("===== E4 验证: 失败模式 + 修复 (free 分支) =====")
b=corr_on(ROOT+"/data/aug.npz",f"{CP}/v40_geoattn/best.pt","微调前 baseline")
a=corr_on(ROOT+"/data/aug.npz",ROOT+"/data/v40_finetuned.pt","微调后 finetuned")
print(f"  free: {b:.3f} -> {a:.3f}  (论文 -0.24 -> -0.99)")
print("===== E4 验证: fixed 分支 =====")
bf=corr_on(ROOT+"/data/aug_fixed.npz",f"{CP}/v40_geoattn/best.pt","微调前 fixed")
af=corr_on(ROOT+"/data/aug_fixed.npz",ROOT+"/data/v40_finetuned_both.pt","微调后 fixed")
print(f"  fixed: {bf:.3f} -> {af:.3f}  (论文 -0.60 -> -0.96)")
print("DONE")
