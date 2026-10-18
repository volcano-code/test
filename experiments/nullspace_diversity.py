"""演示: 同剪影下多候选 —— 剪影IoU都高, 但3D体积差异大(零空间多样性)."""
import sys
from pathlib import Path
import numpy as np,torch
GEN=Path("/home/lin/xvxinyao/ShadowForge/generation")
sys.path.insert(0,str(GEN.parent));sys.path.insert(0,str(GEN))
from scipy.ndimage import binary_erosion,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import (calibrate_tfs_projection,build_tfs_visual_hull,evaluate_tfs_volume,
    connectivity_preserving_carve,minimize_material_printable)
s6=generate_binary_structure(3,1)
B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";R=128
files=[f"{B3}/birds_owl.png",f"{B3}/bugs_ant.png",f"{B3}/reptiles_turtle.png"]
g,_=load_data_from_list(files,True,(R,R),torch.device("cpu"),3,False)
sils=[np.asarray(x).squeeze().astype(np.float32) for x in g]
cal=calibrate_tfs_projection(sils,search_res=96,fix_axis=False)
hull,_=build_tfs_visual_hull(sils,R,calibration=cal)
# 多个零空间候选: 外壳 / 两档腐蚀壳 / 最小可打印体
cands={}
cands["hull(外壳)"]=hull
cands["shell-1(腐蚀1)"]=hull&~binary_erosion(hull,s6,iterations=3)
cands["shell-2(腐蚀2)"]=hull&~binary_erosion(hull,s6,iterations=6)
cands["minimal(最小体)"],_=minimize_material_printable(hull,bridge_radius=1)
def sil_iou(v): return evaluate_tfs_volume(v.astype(np.float32),sils,cal,image_size=256)['mean_iou']
def vol_iou(a,b):
    a=a>0.5;b=b>0.5;u=(a|b).sum();return (a&b).sum()/u if u else 1.0
names=list(cands); occ={k:int(v.sum()) for k,v in cands.items()}
print("=== 各候选: 剪影IoU(对目标) 与 体素占用 ===")
sIoU={k:sil_iou(v) for k,v in cands.items()}
for k in names: print(f"  {k:16s} 剪影IoU={sIoU[k]:.3f}  occ={occ[k]:>6}  材料={occ[k]/occ['hull(外壳)']*100:>5.1f}%hull")
print("\n=== 候选间 3D 体积IoU(差异度: 越低越不同) ===")
print("            "+ "".join(f"{n[:8]:>10}" for n in names))
for a in names:
    print(f"{a[:11]:11s}"+"".join(f"{vol_iou(cands[a],cands[b]):>10.2f}" for b in names))
print(f"\n材料跨度(hull/minimal) = {occ['hull(外壳)']/occ['minimal(最小体)']:.1f}×")
print("DONE")
