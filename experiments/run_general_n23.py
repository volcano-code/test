"""n=2,3 x fixed/free x mr=0,1 严格评估(泛化TSF, 256渲染固定朝向). 生成消融, 无结构SF.
验证锚点: n=3 free+mr=1 应 = 官方 0.994, n=3 fixed+mr=1 = 官方 0.977."""
import sys,json
from pathlib import Path
import numpy as np,torch
GEN=Path("/home/lin/xvxinyao/ShadowForge/generation")
sys.path.insert(0,str(GEN.parent));sys.path.insert(0,str(GEN))
from scipy.ndimage import label,generate_binary_structure
from datasets import load_data_from_list
from tsf_general import build_hull_k,eval_k,search_calib
from voxel_ops import minimize_material_printable,make_watertight_printable
s6=generate_binary_structure(3,1)
B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";R=128;E=256
BASE=["birds_owl","bugs_ant","reptiles_turtle"]
rows=[]
for n in (2,3):
    files=[f"{B3}/{BASE[i]}.png" for i in range(n)]
    g,_=load_data_from_list(files,True,(R,R),torch.device("cpu"),n,False)    # mirror=True: 2k 掩码
    allm=[np.asarray(x).squeeze().astype(np.float32) for x in g]
    sils=allm[:n]; msils=allm[n:2*n] if len(allm)>=2*n else None             # 真实镜像目标
    for branch in ("fixed","free"):
        for mr in (0,1):
            perm,ops=search_calib(sils,n,branch=="fixed",mr,Rs=64,msils=msils)
            H=build_hull_k(sils,perm,ops,R)
            ih=eval_k(H,sils,perm,ops,mr,E,msils)
            v,info=minimize_material_printable(H,bridge_radius=1);vw,_=make_watertight_printable(v,H)
            ic=eval_k(vw,sils,perm,ops,mr,E,msils)
            r=dict(n=n,branch=branch,mr=mr,iou_hull=round(ih,4),iou_carve=round(ic,4),
                   saving=round(info['material_saved_ratio']*100,1),single=int(label(vw,structure=s6)[1]==1))
            rows.append(r);print(r,flush=True)
print("\n===== 泛化TSF 严格(256) n=2,3 x fixed/free x mr =====")
print(f"{'n':>2} {'branch':>6} {'mr':>3} {'IoU_hull':>9} {'IoU_carve':>10} {'省材%':>7} {'单体':>5}")
for r in rows:
    print(f"{r['n']:>2} {r['branch']:>6} {r['mr']:>3} {r['iou_hull']:>9.4f} {r['iou_carve']:>10.4f} {r['saving']:>7.1f} {r['single']:>5}")
json.dump(rows,open(GEN/"output/general_n23_results.json","w"),indent=2)
print("DONE")
