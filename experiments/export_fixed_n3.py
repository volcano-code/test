"""n=3 fixed 分支多样性: 与 free 完全相同的5三元组. 修复: evaluate_tfs_volume 传全部6掩码."""
import sys,json
from pathlib import Path
import numpy as np,torch
GEN=Path("/home/lin/xvxinyao/ShadowForge/generation")
sys.path.insert(0,str(GEN.parent));sys.path.insert(0,str(GEN))
from scipy.ndimage import label,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import (calibrate_tfs_projection,build_tfs_visual_hull,evaluate_tfs_volume,
    minimize_material_printable,make_watertight_printable,sdf_watertight_mesh)
s6=generate_binary_structure(3,1)
B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";R=128;E=256;MM=210.0
OUT=Path("/home/lin/xvxinyao/ShadowForge/print_ready_stls_diversity")
TRI=[("x_dragon","x_peacock","x_trex"),("x_dragon","x_lobster","x_scorpion"),
     ("reptiles_frog","bugs_cricket","x_lobster"),("x_ladybug","x_spider","bugs_cricket"),
     ("bugs_ant","x_dragon","sea_shrimp")]
def vol_iou(a,b):a=a>0.5;b=b>0.5;u=(a|b).sum();return float((a&b).sum()/u) if u else 1.0
def save(vox,name):
    m=sdf_watertight_mesh(vox.astype(np.uint8),smooth_sigma=0.6)
    ext=(m.vertices.max(0)-m.vertices.min(0)).max();m.vertices*=(MM/ext);m.export(str(OUT/f"{name}.stl"));return m.is_watertight
groups=[]
for gi,imn in enumerate(TRI,1):
    try:
        g,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(R,R),torch.device("cpu"),3,False)
        allm=[np.asarray(x).squeeze().astype(np.float32) for x in g]            # 6 掩码(含镜像)
        cal=calibrate_tfs_projection(allm,search_res=96,fix_axis=True)          # fixed
        hull,_=build_tfs_visual_hull(allm,R,calibration=cal)
        mn,_=minimize_material_printable(hull,bridge_radius=1)
        hw,_=make_watertight_printable(hull,hull);mw,_=make_watertight_printable(mn,hull)
        tag="_".join(imn);wt=save(hw,f"g{gi}_{tag}_FIXED_HULL") and save(mw,f"g{gi}_{tag}_FIXED_MINIMAL")
        ih=evaluate_tfs_volume(hw.astype(np.float32),allm,cal,image_size=E)['mean_iou']     # 6掩码
        im=evaluate_tfs_volume(mw.astype(np.float32),allm,cal,image_size=E)['mean_iou']
        rec=dict(n=3,group=gi,triple=tag,branch="fixed",sil_iou_hull=round(float(ih),3),
                 sil_iou_minimal=round(float(im),3),material_ratio=round(hw.sum()/max(mw.sum(),1),1),
                 vol_iou_hull_vs_min=round(vol_iou(hw,mw),3),watertight=wt)
        groups.append(rec)
        print(f"g{gi} {tag} fixed: 剪影IoU hull={rec['sil_iou_hull']}/min={rec['sil_iou_minimal']} "
              f"材料{rec['material_ratio']}× 体积IoU={rec['vol_iou_hull_vs_min']} 水密={wt}",flush=True)
    except Exception as e: print(f"skip g{gi} {imn}: {type(e).__name__} {e}",flush=True)
json.dump(groups,open(OUT/"diversity_metrics_fixed_n3.json","w"),indent=2)
print(f"导出 n=3 fixed {len(groups)} 组 x2 STL");print("DONE")
