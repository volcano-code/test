import sys, json
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import zoom, binary_erosion, generate_binary_structure
SF=Path("/home/lin/xvxinyao/ShadowForge")
for p in [SF, SF/"generation", SF/"experiments"]: sys.path.insert(0,str(p))
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from candidate_selection_batch import make_candidates, keep_largest, n6
S6=generate_binary_structure(3,1); B=str(SF/"data/silhouettes/bench3")
EX={"lobster_bat_frog":["x_lobster","x_bat","reptiles_frog"],"owl_frog_dragon":["birds_owl","reptiles_frog","x_dragon"],
    "spider_dragon_owl":["x_spider","x_dragon","birds_owl"],"eagle_bee_frog":["birds_eagle","x_bee","reptiles_frog"],
    "turtle_peacock_bat":["reptiles_turtle","x_peacock","x_bat"]}
def load(tri,R):
    s,_=load_data_from_list([f'{B}/{n}.png' for n in tri],True,(R,R),torch.device('cpu'),3,False)
    return [np.asarray(x).squeeze().astype(np.float32) for x in s]
rows=[]
print(f'{"example":20s} {"coarse48 IoU":>12s} {"decoupled IoU":>13s} {"省材":>5s} {"单体":>4s} {"水密":>4s}')
for name,tri in EX.items():
    m48=load(tri,48); c48,_=make_candidates(m48,48,True); C48=keep_largest(c48['watertight_export']).astype(bool)
    m256=load(tri,256); c256,a=make_candidates(m256,256,True); H=c256['visual_hull'].astype(bool); cal=a['calibration']; gt6=m256+[np.fliplr(x) for x in m256]; hv=int(H.sum())
    C48u=zoom(C48.astype(np.float32),256/48,order=0)>0.5
    if C48u.shape!=(256,)*3:
        z=np.zeros((256,)*3,bool); sl=tuple(slice(0,min(256,s)) for s in C48u.shape); z[sl]=C48u[sl]; C48u=z
    C48u&=H; shell=H&~binary_erosion(H,S6,1); V=keep_largest(C48u|shell)
    iou_c=evaluate_tfs_volume(C48u.astype(np.float32),gt6,cal,image_size=256)['mean_iou']
    iou_d=evaluate_tfs_volume(V.astype(np.float32),gt6,cal,image_size=256)['mean_iou']
    try: wt=bool(sdf_watertight_mesh(V.astype(np.uint8),smooth_sigma=0.6).is_watertight)
    except: wt=False
    sb=n6(V)==1; sav=1-int(V.sum())/hv
    rows.append(dict(example=name,coarse_iou=round(float(iou_c),4),decoupled_iou=round(float(iou_d),4),saving=round(float(sav),4),single=bool(sb),watertight=bool(wt)))
    print(f'{name:20s} {iou_c:12.3f} {iou_d:13.3f} {sav*100:4.0f}% {str(sb):>4s} {str(wt):>4s}',flush=True)
import numpy as np
cs=[r['coarse_iou'] for r in rows]; ds=[r['decoupled_iou'] for r in rows]
print(f'\n均值: coarse {np.mean(cs):.3f} -> decoupled {np.mean(ds):.3f}  (Δ +{np.mean(ds)-np.mean(cs):.3f})')
print(f'单体 {sum(r["single"] for r in rows)}/{len(rows)}, 水密 {sum(r["watertight"] for r in rows)}/{len(rows)}')
json.dump(rows,open(SF/"experiments/harness/out/decouple_gallery.json","w"),indent=2)
print("DONE")
