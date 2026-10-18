"""n=2 同剪影 hull+minimal 多样性 demo -> 水密STL(210mm). 几何 k=2 轴 TSF(free)."""
import sys,glob,random,json
from pathlib import Path
import numpy as np,torch
GEN=Path("/home/lin/xvxinyao/ShadowForge/generation")
sys.path.insert(0,str(GEN.parent));sys.path.insert(0,str(GEN))
from scipy.ndimage import label,generate_binary_structure
from datasets import load_data_from_list
from tsf_general import build_hull_k,eval_k,search_calib
from voxel_ops import minimize_material_printable,make_watertight_printable,sdf_watertight_mesh
s6=generate_binary_structure(3,1)
B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";R=128;E=256;MM=210.0
OUT=Path("/home/lin/xvxinyao/ShadowForge/print_ready_stls_diversity");OUT.mkdir(exist_ok=True)
imgs=sorted(b.split('/')[-1][:-4] for b in glob.glob(f"{B3}/*.png"))
random.seed(7);pairs=[random.sample(imgs,2) for _ in range(12)]
def vol_iou(a,b):a=a>0.5;b=b>0.5;u=(a|b).sum();return float((a&b).sum()/u) if u else 1.0
def save(vox,name):
    m=sdf_watertight_mesh(vox.astype(np.uint8),smooth_sigma=0.6)
    ext=(m.vertices.max(0)-m.vertices.min(0)).max();m.vertices*=(MM/ext)
    m.export(str(OUT/f"{name}.stl"));return m.is_watertight
groups=[];done=0
for imn in pairs:
    if done>=3: break
    tag="_".join(imn)
    try:
        g,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(R,R),torch.device("cpu"),2,False)
        allm=[np.asarray(x).squeeze().astype(np.float32) for x in g];sils=allm[:2];msils=allm[2:4] if len(allm)>=4 else None
        perm,ops=search_calib(sils,2,False,1,Rs=64,msils=msils)              # n=2, free
        hull=build_hull_k(sils,perm,ops,R)
        if hull.sum()<3000: continue
        mn,_=minimize_material_printable(hull,bridge_radius=1)
        hw,_=make_watertight_printable(hull,hull);mw,_=make_watertight_printable(mn,hull)
        wt=save(hw,f"n2_g{done+1}_{tag}_HULL") and save(mw,f"n2_g{done+1}_{tag}_MINIMAL")
        rec=dict(n=2,group=done+1,pair=tag,
                 sil_iou_hull=round(eval_k(hw,sils,perm,ops,1,E,msils),3),
                 sil_iou_minimal=round(eval_k(mw,sils,perm,ops,1,E,msils),3),
                 material_ratio=round(hw.sum()/max(mw.sum(),1),1),
                 vol_iou_hull_vs_min=round(vol_iou(hw,mw),3),
                 single_body_min=int(label(mw,s6)[1]==1),watertight=wt)
        groups.append(rec);done+=1
        print(f"n2组{done} {tag}: 剪影IoU hull={rec['sil_iou_hull']}/min={rec['sil_iou_minimal']} "
              f"材料{rec['material_ratio']}× 体积IoU={rec['vol_iou_hull_vs_min']} 水密={wt}",flush=True)
    except Exception as e: print(f"skip {tag}: {type(e).__name__} {e}",flush=True)
json.dump(groups,open(OUT/"diversity_metrics_n2.json","w"),indent=2)
print(f"\n导出 {len(groups)} 组 n=2 x2 = {2*len(groups)} STL");print("DONE")
