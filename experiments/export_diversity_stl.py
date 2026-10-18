"""5组 同剪影 hull+minimal 多样性 demo -> 水密STL(210mm). 每组: 同剪影,两个差异极大的实体."""
import sys,glob,random,json
from pathlib import Path
import numpy as np,torch
GEN=Path("/home/lin/xvxinyao/ShadowForge/generation")
sys.path.insert(0,str(GEN.parent));sys.path.insert(0,str(GEN))
from scipy.ndimage import label,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import (calibrate_tfs_projection,build_tfs_visual_hull,evaluate_tfs_volume,
    minimize_material_printable,make_watertight_printable,sdf_watertight_mesh)
s6=generate_binary_structure(3,1)
B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";R=128;MM=210.0
OUT=Path("/home/lin/xvxinyao/ShadowForge/print_ready_stls_diversity");OUT.mkdir(exist_ok=True)
imgs=sorted(b.split('/')[-1][:-4] for b in glob.glob(f"{B3}/*.png"))
random.seed(11);triples=[random.sample(imgs,3) for _ in range(15)]
def silIoU(v,sils,cal): return evaluate_tfs_volume(v.astype(np.float32),sils,cal,image_size=256)['mean_iou']
def vol_iou(a,b):a=a>0.5;b=b>0.5;u=(a|b).sum();return float((a&b).sum()/u) if u else 1.0
def save(vox,hull,name):
    m=sdf_watertight_mesh(vox.astype(np.uint8),smooth_sigma=0.6)
    ext=(m.vertices.max(0)-m.vertices.min(0)).max(); m.vertices*=(MM/ext)        # 缩放到210mm
    m.export(str(OUT/f"{name}.stl")); return m.is_watertight
groups=[];done=0
for imn in triples:
    if done>=5: break
    tag="_".join(imn)
    try:
        g,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(R,R),torch.device("cpu"),3,False)
        sils=[np.asarray(x).squeeze().astype(np.float32) for x in g]
        cal=calibrate_tfs_projection(sils,search_res=96,fix_axis=False)        # free
        hull,_=build_tfs_visual_hull(sils,R,calibration=cal)
        if hull.sum()<3000: continue
        mn,info=minimize_material_printable(hull,bridge_radius=1)
        hw,_=make_watertight_printable(hull,hull); mw,_=make_watertight_printable(mn,hull)
        wt_h=save(hw,hull,f"g{done+1}_{tag}_HULL"); wt_m=save(mw,hull,f"g{done+1}_{tag}_MINIMAL")
        rec=dict(group=done+1,triple=tag,
                 sil_iou_hull=round(silIoU(hw,sils,cal),3),sil_iou_minimal=round(silIoU(mw,sils,cal),3),
                 occ_hull=int(hw.sum()),occ_minimal=int(mw.sum()),
                 material_ratio=round(hw.sum()/max(mw.sum(),1),1),
                 vol_iou_hull_vs_min=round(vol_iou(hw,mw),3),
                 single_body_min=int(label(mw,s6)[1]==1),watertight=(wt_h and wt_m))
        groups.append(rec);done+=1
        print(f"组{done} {tag}: 剪影IoU hull={rec['sil_iou_hull']}/min={rec['sil_iou_minimal']} "
              f"材料{rec['material_ratio']}× 体积IoU={rec['vol_iou_hull_vs_min']} 水密={rec['watertight']}",flush=True)
    except Exception as e: print(f"skip {tag}: {type(e).__name__} {e}",flush=True)
json.dump(groups,open(OUT/"diversity_metrics.json","w"),indent=2)
print(f"\n导出 {len(groups)} 组 x2 = {2*len(groups)} STL -> {OUT}")
print("DONE")
