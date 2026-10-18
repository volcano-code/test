"""Spider-Man/superman/batman 走几何管线: fixed+free 严格IoU + hull/minimal 多样性 STL."""
import sys,json,shutil
from pathlib import Path
import numpy as np,torch
GEN=Path("/home/lin/xvxinyao/ShadowForge/generation")
sys.path.insert(0,str(GEN.parent));sys.path.insert(0,str(GEN))
from scipy.ndimage import label,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import (calibrate_tfs_projection,build_tfs_visual_hull,evaluate_tfs_volume,
    minimize_material_printable,make_watertight_printable,sdf_watertight_mesh)
s6=generate_binary_structure(3,1);R=128;E=256;MM=210.0
HEROES=["Spider-Man","superman","batman"]
OUT=Path("/home/lin/xvxinyao/ShadowForge/print_ready_stls_diversity/by_group/n3_heroes_spiderman_superman_batman")
OUT.mkdir(parents=True,exist_ok=True);(OUT/"silhouettes").mkdir(exist_ok=True)
def vol_iou(a,b):a=a>0.5;b=b>0.5;u=(a|b).sum();return float((a&b).sum()/u) if u else 1.0
def save(vox,name):
    m=sdf_watertight_mesh(vox.astype(np.uint8),smooth_sigma=0.6)
    ext=(m.vertices.max(0)-m.vertices.min(0)).max();m.vertices*=(MM/ext);m.export(str(OUT/f"{name}.stl"));return m.is_watertight
files=[str(GEN/f"{h}.png") for h in HEROES]
for h in HEROES: shutil.copy(GEN/f"{h}.png",OUT/"silhouettes"/f"{h}.png")
g,_=load_data_from_list(files,True,(R,R),torch.device("cpu"),3,False)
allm=[np.asarray(x).squeeze().astype(np.float32) for x in g]                 # 6 掩码
met={"group":"n3_heroes_spiderman_superman_batman","n":3,"silhouettes":HEROES}
for branch,fa,pre in [("free",False,"heroes_FREE"),("fixed",True,"heroes_FIXED")]:
    cal=calibrate_tfs_projection(allm,search_res=96,fix_axis=fa)
    hull,_=build_tfs_visual_hull(allm,R,calibration=cal)
    mn,_=minimize_material_printable(hull,bridge_radius=1)
    hw,_=make_watertight_printable(hull,hull);mw,_=make_watertight_printable(mn,hull)
    wt=save(hw,f"{pre}_HULL") and save(mw,f"{pre}_MINIMAL")
    ih=evaluate_tfs_volume(hw.astype(np.float32),allm,cal,image_size=E)['mean_iou']
    im=evaluate_tfs_volume(mw.astype(np.float32),allm,cal,image_size=E)['mean_iou']
    met[branch]=dict(sil_iou_hull=round(float(ih),3),sil_iou_minimal=round(float(im),3),
                     material_ratio=round(hw.sum()/max(mw.sum(),1),1),vol_iou_hull_vs_min=round(vol_iou(hw,mw),3),
                     single_body_min=int(label(mw,s6)[1]==1),watertight=wt)
    print(f"{branch}: 剪影IoU hull={met[branch]['sil_iou_hull']}/min={met[branch]['sil_iou_minimal']} "
          f"材料{met[branch]['material_ratio']}× 体积IoU={met[branch]['vol_iou_hull_vs_min']} 水密={wt}",flush=True)
json.dump(met,open(OUT/"metrics.json","w"),indent=2,ensure_ascii=False)
print(f"导出 -> {OUT} (4 STL + 3 剪影 + metrics)");print("DONE")
