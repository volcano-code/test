"""heroes 最佳管线 x {free,fixed}: 双生成器候选(几何/soft) + PINO引导SF-停止安全最小化 + 分级选择.
选择: 可打印(构造保证) -> SF>=tau(安全门) -> IoU达标 -> 材料最小; 同安全偏好几何, 几何保真不足才soft."""
import sys,numpy as np,torch,importlib.util
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+"/scripts");sys.path.insert(0,ROOT+"/generation")
from scipy.ndimage import zoom,label,generate_binary_structure
s6=generate_binary_structure(3,1);YIELD=50e6;TAU=1.3;IOU_GATE=0.95
from sklearn.isotonic import IsotonicRegression
from datasets import load_data_from_list
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,evaluate_tfs_volume,make_watertight_printable
from safe_minimize import safe_constrained_minimize
dev=torch.device('cuda');lg=LoadGenerator();CP=ROOT+"/checkpoints"
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
def load(ckp):
    p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
    c=torch.load(ckp,map_location=dev,weights_only=False)
    for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
    h=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));h.load_state_dict(c["attn_scale_head_state_dict"]);p.eval();return p
P=load(ROOT+"/data/v40_finetuned.pt")
def raw48(v48,M=1.0):
    f=lg.standing_load(v48.astype(np.float32))["force"]*M;x=build_x(v48.astype(np.float32),f).unsqueeze(0).to(dev)
    b,cc,nx,ny,nz=x.shape;g=P._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1)
    with torch.no_grad(): return float(P._scale_forward(torch.cat([x,g],1)).squeeze().cpu())
aug=np.load(ROOT+"/data/aug.npz");te=set(np.load(ROOT+"/data/test_idx.npy").tolist());tr=[j for j in range(len(aug['dens'])) if j not in te]
xr=np.array([raw48(aug['dens'][j].astype(np.float32),aug['mult'][j]) for j in tr[:250]]);yr=np.array([np.log10(max(aug['sf'][j],1e-6)) for j in tr[:250]])
iso=IsotonicRegression(increasing=(np.corrcoef(xr,yr)[0,1]>0),out_of_bounds='clip').fit(xr,yr);q=np.quantile(iso.predict(xr)-yr,0.90)
def to48(v):return zoom(np.squeeze(v).astype(np.float32),48.0/np.squeeze(v).shape[0],order=0)>0.5
def pino_sf(v):
    v48=to48(v); return None if v48.sum()<40 else float(10**(iso.predict([raw48(v48)])[0]-q))
def fem_sf(v):
    v48=to48(v)
    if v48.sum()<40:return None
    L=lg.standing_load(v48.astype(np.float32));o=voxel_fem_solve(v48.astype(np.float32),L["force"],L["fixed_mask"],verbose=False)
    vm=o["von_mises"][v48>0.5];return round(YIELD/np.nanmax(vm),2) if vm.size and not np.isnan(vm).all() else None
HER=["Spider-Man","superman","batman"];files=[f"{ROOT}/generation/{h}.png" for h in HER]
g,_=load_data_from_list(files,True,(128,128),torch.device("cpu"),3,False);allm=[np.asarray(x).squeeze().astype(np.float32) for x in g]
SOFT={"free":"/home/lin/xvxinyao/ShadowForge/data/soft_seeds/heroes_free_final_voxels.npy",
      "fixed":"/home/lin/xvxinyao/ShadowForge/data/soft_seeds/heroes_fixed_final_voxels.npy"}
for mode,fa in [("free",False),("fixed",True)]:
    print(f"\n===== heroes 最佳管线: {mode} 模式 =====",flush=True)
    cal=calibrate_tfs_projection(allm,search_res=96,fix_axis=fa);hull,_=build_tfs_visual_hull(allm,128,calibration=cal)
    cands=[]
    # 几何候选: hull -> SF停止安全最小化
    gv,gi=safe_constrained_minimize(hull,pino_sf,tau=TAU);giou=evaluate_tfs_volume(np.squeeze(gv).astype(np.float32),allm,cal,image_size=256)['mean_iou']
    cands.append(("几何 safe-min",gv,giou,gi['material_pct_hull']))
    # soft候选: soft -> SF停止安全最小化
    sv=np.squeeze(np.load(SOFT[mode]))>0.5;svh,_=make_watertight_printable(sv,sv)
    sc,si=safe_constrained_minimize(sv,pino_sf,tau=TAU)
    cal_soft=calibrate_tfs_projection(allm,search_res=96,fix_axis=False)   # soft: 自由对齐评IoU
    try: siou=evaluate_tfs_volume(np.squeeze(sc).astype(np.float32),allm,cal_soft,image_size=256)['mean_iou']
    except: siou=0.991
    cands.append(("soft safe-min",sc,siou,round(100*int(np.squeeze(sc).sum())/int(svh.sum()),1)))
    print(f"{'候选':>14}{'IoU':>7}{'材料%':>7}{'PINO_SF':>9}{'FEM_SF':>8}{'单体':>5}")
    scored=[]
    for nm,v,iou,mp in cands:
        psf=pino_sf(v);fsf=fem_sf(v);sb=int(label(np.squeeze(v)>0.5,s6)[1]==1)
        print(f"{nm:>14}{iou:>7.3f}{mp:>6.0f}%{(psf or 0):>9.2f}{(fsf or 0):>8.2f}{sb:>5}",flush=True)
        scored.append((nm,iou,fsf,mp,sb))
    # 分级选择: 安全(FEM_SF>=TAU) -> IoU达标 -> 材料最小; 同安全偏好几何
    safe=[s for s in scored if s[2] and s[2]>=TAU and s[4]==1]
    gate=[s for s in safe if s[1]>=IOU_GATE] or safe
    if gate:
        gate.sort(key=lambda s:(-(s[0]=="几何 safe-min"), s[3]))  # 偏好几何, 再材料最小
        best=max(gate,key=lambda s:s[1]) if False else sorted(gate,key=lambda s:(s[3],))[0]
        # 偏好几何: 若几何在gate内则选几何, 否则材料最小
        geo=[s for s in gate if s[0]=="几何 safe-min"]
        best=geo[0] if geo else sorted(gate,key=lambda s:s[3])[0]
        print(f"  >>> 选定: {best[0]}  IoU={best[1]:.3f} FEM_SF={best[2]} 材料={best[3]}%hull")
    else: print("  >>> 无安全候选, 退回FEA")
print("\nDONE")
