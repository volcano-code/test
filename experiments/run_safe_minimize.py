"""演示 PINO引导 SF-停止安全最小化(soft-free英雄). PINO=v40_finetuned, ACS isotonic 映射 raw->SF. 交叉验FEM."""
import sys,numpy as np,torch,importlib.util
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+"/scripts");sys.path.insert(0,ROOT+"/generation")
from scipy.ndimage import zoom
from sklearn.isotonic import IsotonicRegression
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
from safe_minimize import safe_constrained_minimize
dev=torch.device('cuda');lg=LoadGenerator();CP=ROOT+"/checkpoints";YIELD=50e6;TAU=1.3
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
def load(ckp):
    p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
    c=torch.load(ckp,map_location=dev,weights_only=False)
    for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
    h=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));h.load_state_dict(c["attn_scale_head_state_dict"]);p.eval();return p
P=load(ROOT+"/data/v40_finetuned.pt")
def raw48(v48,M=1.0):
    f=lg.standing_load(v48.astype(np.float32))["force"]*M
    x=build_x(v48.astype(np.float32),f).unsqueeze(0).to(dev);b,cc,nx,ny,nz=x.shape
    g=P._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1)
    with torch.no_grad(): return float(P._scale_forward(torch.cat([x,g],1)).squeeze().cpu())
# ACS: 在 aug 上拟合 raw->log_SF(isotonic, 递减), 取保守(单边)
aug=np.load(ROOT+"/data/aug.npz");te=set(np.load(ROOT+"/data/test_idx.npy").tolist())
tr=[j for j in range(len(aug['dens'])) if j not in te]
xr=[];yr=[]
for j in tr[:300]:
    xr.append(raw48(aug['dens'][j].astype(np.float32),aug['mult'][j]));yr.append(np.log10(max(aug['sf'][j],1e-6)))
xr=np.array(xr);yr=np.array(yr)
iso=IsotonicRegression(increasing=(np.corrcoef(xr,yr)[0,1]>0),out_of_bounds='clip').fit(xr,yr)
resid=iso.predict(xr)-yr; q=np.quantile(resid,0.90)        # 单边保守
def pino_sf(v128):
    v48=zoom(np.squeeze(v128).astype(np.float32),48.0/np.squeeze(v128).shape[0],order=0)>0.5
    if v48.sum()<40: return None
    return float(10**(iso.predict([raw48(v48)])[0]-q))      # ACS 保守 SF
def fem_sf(v128):
    v48=zoom(np.squeeze(v128).astype(np.float32),48.0/np.squeeze(v128).shape[0],order=0)>0.5
    if v48.sum()<40: return None
    o=voxel_fem_solve(v48.astype(np.float32),lg.standing_load(v48.astype(np.float32))["force"],lg.standing_load(v48.astype(np.float32))["fixed_mask"],verbose=False)
    vm=o["von_mises"][v48>0.5];return float(YIELD/np.nanmax(vm)) if vm.size and not np.isnan(vm).all() else None
print(f"ACS拟合: corr(raw,SF)={np.corrcoef(xr,yr)[0,1]:.2f} 保守q={q:.3f}")
sv=np.squeeze(np.load("/home/lin/xvxinyao/ShadowForge/data/soft_seeds/heroes_free_final_voxels.npy"))>0.5
out,info=safe_constrained_minimize(sv,pino_sf,tau=TAU,margin=1.0,max_steps=8)
print("\nPINO引导 SF-停止 安全最小化(soft-free英雄):")
print(f"  阶梯(occ, PINO保守SF): {[(o,round(s,2) if s else None) for o,s in info['trace']]}")
print(f"  选定: 材料={info['material_pct_hull']}%hull  PINO保守SF={round(info['predicted_sf'],2)}  FEM真值SF={round(fem_sf(out),2)}")
print(f"  对比盲目minimal: FEM SF={round(fem_sf(sv if False else __import__('voxel_ops').make_watertight_printable(__import__('voxel_ops').minimize_material_printable(sv)[0],sv)[0]),2)}")
print("DONE")
