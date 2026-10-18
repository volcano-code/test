#!/usr/bin/env python3
"""生成 FIXED 分支的可打印交付件(供物理打印 fixed 例)。
fix_axis=True 候选池, 过 select_final 同硬约束(单体∧水密∧IoU≥0.95∧dual-ACS+FEM安全∧壁厚erosion≥0.15),
实心集内最大省材。导出 STL + 各方向剪影@512 + 输入。"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from run_all_baselines import evaluate_metrics, load_pino
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from structural.analysis.load_generator import LoadGenerator
from candidate_selection_batch import make_candidates, fit_acs, fem_sf, n6, select_final, TAU
_e=importlib.util.spec_from_file_location("ehqm",str(ROOT/"generation/export_high_quality_mesh.py")); ehqm=importlib.util.module_from_spec(_e); _e.loader.exec_module(ehqm)
import trimesh
_S6=generate_binary_structure(3,1)
def ero(v): return float(binary_erosion(v,_S6).sum()/max(int(v.sum()),1))
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
pino=load_pino(str(ROOT/"checkpoints/pino_v24_ufno_dual/best_model.pt"),dev)
iso,q_safe,q_unsafe=fit_acs(0.10); tau_log=float(np.log10(TAU)); R=48
def save_white(m,p): Image.fromarray((np.asarray(m)>0.5).astype(np.uint8)*255).resize((512,512),Image.NEAREST).save(p)
B=str(ROOT/"data/silhouettes/bench3"); H=str(ROOT/"generation")
EX={"heroes_spiderman_superman_batman":[f"{H}/Spider-Man.png",f"{H}/superman.png",f"{H}/batman.png"],
 "dragon_peacock_trex":[f"{B}/x_dragon.png",f"{B}/x_peacock.png",f"{B}/x_trex.png"],
 "owl_bat_snail":[f"{B}/birds_owl.png",f"{B}/x_bat.png",f"{B}/sea_snail.png"]}
OUT=ROOT/"results/gallery_FINAL_fixed"; OUT.mkdir(parents=True,exist_ok=True); man=[]
for name,paths in EX.items():
    t0=time.time()
    s,_=load_data_from_list(paths,True,(R,R),torch.device("cpu"),3,False); m=[np.asarray(x).squeeze().astype(np.float32) for x in s]
    cands,aux=make_candidates(m,R,True); hull=aux["hull"]; cal=aux["calibration"]; hv=int(hull.sum()); gt6=m+[np.fliplr(x) for x in m]
    rows=[]
    for cn,vol in cands.items():
        iou=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=R)["mean_iou"]
        try: wt=bool(sdf_watertight_mesh(vol,smooth_sigma=0.6).is_watertight)
        except: wt=False
        load=LoadGenerator().standing_load(vol.astype(np.float32))
        sfp=float(evaluate_metrics(vol.astype(np.float32),hull.astype(np.float32),load["force"],pino,dev)["sf_min_pino"])
        pred=float(iso.predict([np.log10(max(sfp,1e-12))])[0]); fe=fem_sf(vol,1.0)
        rows.append({"candidate":cn,"branch":"fixed","mean_iou":round(iou,4),"component_count_6":n6(vol),"watertight":wt,
            "material_saved_vs_hull":round(1-int(vol.sum())/max(hv,1),4),"occupied_voxels":int(vol.sum()),"erosion_survival":round(ero(vol),3),
            "sf_pino":round(sfp,3),"acs_certified_safe":bool(pred-q_safe>=tau_log),"acs_certified_unsafe":bool(pred+q_unsafe<tau_log),
            "fem_safe":fe["safe"],"sf_fem":round(fe["sf_fem"],3),"_vol":vol})
    chosen,info=select_final(rows)
    d=OUT/name; (d/"inputs").mkdir(parents=True,exist_ok=True); (d/"silhouettes").mkdir(parents=True,exist_ok=True)
    json.dump([{k:v for k,v in r.items() if not k.startswith("_")} for r in rows],open(d/"candidate_scores.json","w"),indent=2,ensure_ascii=False)
    for i,p in enumerate(paths):
        g=np.array(Image.open(p).convert("L")); Image.fromarray((g>127).astype(np.uint8)*255).resize((512,512),Image.LANCZOS).save(d/"inputs"/f"input{i}.png")
    if chosen is None:
        man.append({"example":name,"status":"DEFER","n_fem_used":info["n_fem_used"]}); print(f"{name}(fixed): DEFER"); continue
    vol=chosen["_vol"]; outp=d/f"{name}_FIXED.stl"
    ehqm.export_marching_cubes(vol.astype(bool),outp,smooth_sigma=0.35,taubin_iterations=2,upsample_factor=4)
    r=trimesh.load(str(outp)); r.merge_vertices(); r.update_faces(r.unique_faces())
    cc=r.split(only_watertight=False)
    if cc: r=max(cc,key=lambda x:len(x.faces))
    trimesh.repair.fill_holes(r)
    if r.volume<0: r.invert()
    trimesh.repair.fix_normals(r); r.export(str(outp)); np.save(d/f"{name}_voxels48.npy",vol.astype(np.uint8))
    ev=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=512)
    for i in range(3): save_white(ev["pred_masks"][i],d/"silhouettes"/f"shadow{i}.png")
    man.append({"example":name,"branch":"fixed","selected":chosen["candidate"],"decided_by":info["decided_by"],
        "mean_iou":chosen["mean_iou"],"material_saved":chosen["material_saved_vs_hull"],"sf_fem":chosen["sf_fem"],
        "mesh_parts":len(r.split(only_watertight=False)),"watertight":bool(r.is_watertight),"sec":round(time.time()-t0)})
    print(f"{name}(fixed): 选 {chosen['candidate']} IoU={chosen['mean_iou']:.3f} 省材={chosen['material_saved_vs_hull']*100:.0f}% SF={chosen['sf_fem']} 认证={info['decided_by']} {time.time()-t0:.0f}s")
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False); print("DONE ->",OUT)
