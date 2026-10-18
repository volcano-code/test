"""最严格: 论文真实管线 + evaluate_tfs_volume(256渲染,固定标定朝向). fixed/free x n=2,3, tsf=1.
fixed=calibrate(fix_axis=True), free=calibrate(fix_axis=False, 含d4镜像搜索)."""
import sys
from pathlib import Path
import numpy as np,torch
GEN=Path("/home/lin/xvxinyao/ShadowForge/generation")
sys.path.insert(0,str(GEN.parent));sys.path.insert(0,str(GEN))
from scipy.ndimage import label,generate_binary_structure,zoom
from datasets import load_data_from_list
from voxel_ops import (calibrate_tfs_projection,build_tfs_visual_hull,evaluate_tfs_volume,
    minimize_material_printable,make_watertight_printable)
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
s6=generate_binary_structure(3,1);YIELD=50e6;TAU=1.3;lg=LoadGenerator()
B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";R=128
BASE=["birds_owl","bugs_ant","reptiles_turtle"]
def binding(vox):
    R0=vox.shape[0];v=zoom(vox.astype(np.float32),48.0/R0,order=0)>0.5
    if v.sum()<40: return None
    L=lg.standing_load(v.astype(np.float32));o=voxel_fem_solve(v.astype(np.float32),L["force"],L["fixed_mask"],verbose=False)
    vm=o["von_mises"][v>0.5]
    if not(vm.size and not np.isnan(vm).all()): return None
    mx=float(np.nanmax(vm));return None if mx<1e-6 else float(YIELD/mx)
rows=[]
for n in (3,):
    files=[f"{B3}/{BASE[i]}.png" for i in range(n)]
    g,_=load_data_from_list(files,True,(R,R),torch.device("cpu"),n,False)
    sils=[np.asarray(x).squeeze().astype(np.float32) for x in g]
    for branch in ("fixed","free"):
        cal=calibrate_tfs_projection(sils,search_res=96,fix_axis=(branch=="fixed"))
        hull,_=build_tfs_visual_hull(sils,R,calibration=cal)
        iou_hull=evaluate_tfs_volume(hull.astype(np.float32),sils,cal,image_size=256)['mean_iou']
        v,info=minimize_material_printable(hull,bridge_radius=1);vw,_=make_watertight_printable(v,hull)
        iou_carve=evaluate_tfs_volume(vw.astype(np.float32),sils,cal,image_size=256)['mean_iou']
        sf=binding(vw)
        r=dict(n=n,branch=branch,iou_hull=round(float(iou_hull),4),iou_carve=round(float(iou_carve),4),
               saving=round(info['material_saved_ratio']*100,1),single=int(label(vw,structure=s6)[1]==1),sf=round(sf,2) if sf else None)
        rows.append(r);print(r,flush=True)
print("\n===== 最严格(evaluate_tfs_volume@256,固定朝向) fixed/free x n (tsf=1) =====")
print(f"{'n':>2} {'branch':>6} {'IoU_hull':>9} {'IoU_carve':>10} {'省材%':>7} {'单体':>5} {'SF':>7}")
for r in rows:
    print(f"{r['n']:>2} {r['branch']:>6} {r['iou_hull']:>9.4f} {r['iou_carve']:>10.4f} {r['saving']:>7.1f} {r['single']:>5} {(r['sf'] or 0):>7.2f}")
print("DONE")
