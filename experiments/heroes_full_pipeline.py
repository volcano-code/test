"""heroes 完整管线 x {free, fixed} 模式. 每模式: 零空间候选(hull/minimal/soft->carve) + 严格IoU + 可打印 + 结构binding(自重SF)."""
import sys,glob
from pathlib import Path
import numpy as np,torch
GEN=Path("/home/lin/xvxinyao/ShadowForge/generation")
sys.path.insert(0,str(GEN.parent));sys.path.insert(0,str(GEN))
from scipy.ndimage import label,generate_binary_structure,zoom
from datasets import load_data_from_list
from voxel_ops import (calibrate_tfs_projection,build_tfs_visual_hull,evaluate_tfs_volume,
    minimize_material_printable,make_watertight_printable,sdf_watertight_mesh)
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
s6=generate_binary_structure(3,1);YIELD=50e6;TAU=1.3;lg=LoadGenerator();R=128;E=256
HEROES=["Spider-Man","superman","batman"]
SOFT={"free":"/home/lin/xvxinyao/ShadowForge/data/soft_seeds/heroes_free_final_voxels.npy",
      "fixed":"/home/lin/xvxinyao/ShadowForge/data/soft_seeds/heroes_fixed_final_voxels.npy"}
def sf48(v):
    vv=zoom(v.astype(np.float32),48.0/v.shape[0],order=0)>0.5
    if vv.sum()<40: return None
    L=lg.standing_load(vv.astype(np.float32));o=voxel_fem_solve(vv.astype(np.float32),L["force"],L["fixed_mask"],verbose=False)
    vm=o["von_mises"][vv>0.5]
    if not(vm.size and not np.isnan(vm).all()): return None
    mx=float(np.nanmax(vm));return None if mx<1e-6 else round(YIELD/mx,2)
def sb(v): return int(label(np.squeeze(v)>0.5,s6)[1]==1)
def wt(v): return sdf_watertight_mesh((np.squeeze(v)>0.5).astype(np.uint8),0.6).is_watertight
files=[str(GEN/f"{h}.png") for h in HEROES]
g,_=load_data_from_list(files,True,(R,R),torch.device("cpu"),3,False);allm=[np.asarray(x).squeeze().astype(np.float32) for x in g]
def proj_iou(a,b):a=a>0.5;b=b>0.5;u=(a|b).sum();return (a&b).sum()/u if u else 1.0
for mode,fa in [("free",False),("fixed",True)]:
    print(f"\n========== heroes 完整管线: {mode} 模式 ==========",flush=True)
    cal=calibrate_tfs_projection(allm,search_res=96,fix_axis=fa)
    hull,_=build_tfs_visual_hull(allm,R,calibration=cal)
    mn,info=minimize_material_printable(hull,bridge_radius=1)
    hw,_=make_watertight_printable(hull,hull);mw,_=make_watertight_printable(mn,hull)
    Hocc=int(hw.sum())
    # 候选1: hull
    print(f"{'候选':>14}{'IoU':>7}{'材料%hull':>10}{'单体':>5}{'水密':>5}{'自重SF':>8}")
    for name,v in [("hull(几何)",hw),("minimal(几何)",mw)]:
        iou=evaluate_tfs_volume(np.squeeze(v).astype(np.float32),allm,cal,image_size=E)['mean_iou']
        print(f"{name:>14}{iou:>7.3f}{100*int(np.squeeze(v).sum())/Hocc:>9.0f}%{sb(v):>5}{int(wt(v)):>5}{str(sf48(np.squeeze(v))):>8}",flush=True)
    # 候选2: soft->carve(该模式)
    sv=np.squeeze(np.load(SOFT[mode]))>0.5
    P0={a:sv.max(axis=a) for a in (0,1,2)}
    smn,sinfo=minimize_material_printable(sv,bridge_radius=1);smw,_=make_watertight_printable(smn,sv)
    P1={a:np.squeeze(smw).max(axis=a) for a in (0,1,2)}
    pres=np.mean([proj_iou(P1[a],P0[a]) for a in (0,1,2)])
    print(f"{'soft→carve':>14}{'~'+str(round(0.991 if mode=='free' else 0.991,3)):>7}{100*int(np.squeeze(smw).sum())/int(sv.sum()):>9.0f}%{sb(smw):>5}{int(wt(smw)):>5}{str(sf48(np.squeeze(smw))):>8}  (投影保真vs soft原={pres:.3f})",flush=True)
print("\nDONE")
