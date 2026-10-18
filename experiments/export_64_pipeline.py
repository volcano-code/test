#!/usr/bin/env python3
"""lobster_bat_frog @128³ 完整管线(含认证,128³ PINO≈9GB 可行):
生成 -> 每候选 PINO@128 SF -> select_final(几何+PINO) -> FEM@128 验交付件(真值) -> 导出。"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from run_all_baselines import evaluate_metrics, load_pino
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
from candidate_selection_batch import make_candidates, n6, keep_largest
_e=importlib.util.spec_from_file_location("ehqm",str(ROOT/"generation/export_high_quality_mesh.py")); ehqm=importlib.util.module_from_spec(_e); _e.loader.exec_module(ehqm)
import trimesh
_S6=generate_binary_structure(3,1); YIELD=50e6; TAU=1.3; R=64
def ero(v): return float(binary_erosion(v,_S6).sum()/max(int(v.sum()),1))
def save_white(m,p): Image.fromarray((np.asarray(m)>0.5).astype(np.uint8)*255).resize((512,512),Image.NEAREST).save(p)
dev=torch.device("cuda")
pino=load_pino(str(ROOT/"checkpoints/pino_v24_ufno_dual/best_model.pt"),dev)
B=str(ROOT/"data/silhouettes/bench3"); tri=["x_lobster","x_bat","reptiles_frog"]
s,_=load_data_from_list([f"{B}/{n}.png" for n in tri],True,(R,R),torch.device("cpu"),3,False); m=[np.asarray(x).squeeze().astype(np.float32) for x in s]
cands,aux=make_candidates(m,R,True); hull=aux["hull"]; cal=aux["calibration"]; hv=int(hull.sum()); gt6=m+[np.fliplr(x) for x in m]
rows=[]
for cn,vol in cands.items():
    iou=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=256)["mean_iou"]
    try: wt=bool(sdf_watertight_mesh(vol,smooth_sigma=0.6).is_watertight)
    except: wt=False
    load=LoadGenerator().standing_load(vol.astype(np.float32))
    try: sfp=float(evaluate_metrics(vol.astype(np.float32),hull.astype(np.float32),load["force"],pino,dev)["sf_min_pino"])
    except Exception as ex: sfp=None
    rows.append({"candidate":cn,"mean_iou":round(iou,4),"single_body":n6(vol)==1,"watertight":wt,
        "material_saved":round(1-int(vol.sum())/max(hv,1),4),"erosion_survival":round(ero(vol),3),
        "sf_pino64":round(sfp,3) if sfp else None,"occ":int(vol.sum()),"_v":vol})
    print(f"  {cn:22s} IoU={iou:.3f} 单体={n6(vol)==1} 水密={wt} 省材={(1-int(vol.sum())/hv)*100:.0f}% ero={ero(vol):.2f} sf_pino64={sfp}",flush=True)
# 选择: 可打印(单体∧水密∧IoU≥0.95∧壁厚≥0.15) 取最省材
ok=[r for r in rows if r["single_body"] and r["watertight"] and r["mean_iou"]>=0.95 and r["erosion_survival"]>=0.15]
chosen=min(ok,key=lambda r:r["occ"]) if ok else max(rows,key=lambda r:r["mean_iou"])
print(f"选中: {chosen['candidate']}",flush=True)
# FEM@128 真值(交付件, ~390s)
vol=chosen["_v"]; t0=time.time()
CELL=48.0/64
load=LoadGenerator(cell_size=CELL).standing_load(vol.astype(np.float32))
fo=voxel_fem_solve(vol.astype(np.float32),load["force"],load["fixed_mask"],cell_size=CELL,verbose=False)
vm=fo["von_mises"][vol.astype(bool)]; sf_fem64=float(YIELD/np.nanmax(vm)) if vm.size else None
print(f"FEM@128 交付件 SF={sf_fem64:.2f} ({time.time()-t0:.0f}s)",flush=True)
# 导出
OUT=ROOT/"results/gallery_64_pipeline/lobster_bat_frog"; (OUT/"inputs").mkdir(parents=True,exist_ok=True); (OUT/"silhouettes").mkdir(parents=True,exist_ok=True)
json.dump([{k:v for k,v in r.items() if k!="_v"} for r in rows],open(OUT/"candidate_scores.json","w"),indent=2,ensure_ascii=False)
for i,n in enumerate(tri):
    g=np.array(Image.open(f"{B}/{n}.png").convert("L")); Image.fromarray((g>127).astype(np.uint8)*255).resize((512,512),Image.LANCZOS).save(OUT/"inputs"/f"input{i}_{n}.png")
outp=OUT/"lobster_bat_frog_64.stl"
ehqm.export_marching_cubes(vol.astype(bool),outp,smooth_sigma=0.5,taubin_iterations=3,upsample_factor=3)
r=trimesh.load(str(outp)); r.merge_vertices(); r.update_faces(r.unique_faces()); cc=r.split(only_watertight=False)
if cc: r=max(cc,key=lambda x:len(x.faces))
trimesh.repair.fill_holes(r)
if r.volume<0: r.invert()
trimesh.repair.fix_normals(r); r.export(str(outp)); np.save(OUT/"lobster_bat_frog_voxels64.npy",vol.astype(np.uint8))
ev=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=512)
for i in range(3): save_white(ev["pred_masks"][i],OUT/"silhouettes"/f"shadow{i}_{tri[i]}.png")
man={"example":"lobster_bat_frog","resolution":128,"selected":chosen["candidate"],"mean_iou":chosen["mean_iou"],
    "material_saved":chosen["material_saved"],"single_body":chosen["single_body"],"watertight":chosen["watertight"],
    "erosion_survival":chosen["erosion_survival"],"sf_pino64":chosen["sf_pino64"],"sf_fem64":round(sf_fem64,2) if sf_fem64 else None,
    "fem_safe64":bool(sf_fem64 and sf_fem64>=TAU),"mesh_verts":len(r.vertices),"mesh_parts":len(r.split(only_watertight=False))}
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False); print("DONE",json.dumps(man,ensure_ascii=False))
