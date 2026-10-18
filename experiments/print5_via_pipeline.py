#!/usr/bin/env python3
"""用权威管线(candidate_selection_batch 的 7候选池 + dual-ACS + FEM兜底 + select)
跑 5 个打印实例(含 heroes), 各导出 dual_acs_fem_fallback 选中件的 STL(256 taubin)."""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, torch
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from run_all_baselines import evaluate_metrics, load_pino
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from structural.analysis.load_generator import LoadGenerator
# 复用权威管线函数
from candidate_selection_batch import make_candidates, fit_acs, select, fem_sf, n6, TAU
_e=importlib.util.spec_from_file_location("ehqm",str(ROOT/"generation/export_high_quality_mesh.py")); ehqm=importlib.util.module_from_spec(_e); _e.loader.exec_module(ehqm)
import trimesh
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
pino=load_pino(str(ROOT/"checkpoints/pino_v24_ufno_dual/best_model.pt"),dev)
iso,q_safe,q_unsafe=fit_acs(0.10); tau_log=float(np.log10(TAU)); RES=48; BRANCH="free"
B=str(ROOT/"data/silhouettes/bench3"); H=str(ROOT/"generation")
EXAMPLES={
 "heroes_spiderman_superman_batman":[f"{H}/Spider-Man.png",f"{H}/superman.png",f"{H}/batman.png"],
 "dragon_peacock_trex":[f"{B}/x_dragon.png",f"{B}/x_peacock.png",f"{B}/x_trex.png"],
 "frog_cricket_lobster":[f"{B}/reptiles_frog.png",f"{B}/bugs_cricket.png",f"{B}/x_lobster.png"],
 "parrot_squid_scorpion":[f"{B}/birds_parrot.png",f"{B}/sea_squid.png",f"{B}/x_scorpion.png"],
 "owl_bat_snail":[f"{B}/birds_owl.png",f"{B}/x_bat.png",f"{B}/sea_snail.png"],
}
OUT=ROOT/"results/print5_pipeline"; OUT.mkdir(parents=True,exist_ok=True)
man=[]
for name,paths in EXAMPLES.items():
    t0=time.time()
    sils,_=load_data_from_list(paths,True,(RES,RES),torch.device("cpu"),3,False)
    masks=[np.asarray(x).squeeze().astype(np.float32) for x in sils]
    cands,aux=make_candidates(masks,RES,BRANCH=="fixed"); hull=aux["hull"]; cal=aux["calibration"]
    rows=[]
    for cn,vol in cands.items():
        m=evaluate_tfs_volume(vol.astype(np.float32),masks,cal,image_size=RES)
        try: wt=bool(sdf_watertight_mesh(vol,smooth_sigma=0.6).is_watertight)
        except: wt=False
        load=LoadGenerator().standing_load(vol.astype(np.float32))
        sf_pino=float(evaluate_metrics(vol.astype(np.float32),hull.astype(np.float32),load["force"],pino,dev)["sf_min_pino"])
        pred=float(iso.predict([np.log10(max(sf_pino,1e-12))])[0]); fe=fem_sf(vol,1.0)
        rows.append({"candidate":cn,"occupied_voxels":int(vol.sum()),"material_saved_vs_hull":1-int(vol.sum())/max(int(hull.sum()),1),
            "mean_iou":m["mean_iou"],"component_count_6":n6(vol),"watertight":wt,"sf_pino":sf_pino,"raw_pino_safe":sf_pino>=TAU,
            "acs_certified_safe":bool(pred-q_safe>=tau_log),"acs_certified_unsafe":bool(pred+q_unsafe<tau_log),
            "sf_fem":fe["sf_fem"],"fem_safe":fe["safe"],"_vol":vol})
    chosen,fb=select(rows,"dual_acs_fem_fallback")
    d=OUT/name; d.mkdir(exist_ok=True)
    if chosen is None:
        man.append({"example":name,"status":"DEFER(无安全可打印候选)","sec":round(time.time()-t0)}); print(f"{name}: DEFER"); continue
    vol=chosen["_vol"]
    ehqm.export_marching_cubes(vol.astype(bool),d/f"{name}_SELECTED.stl",smooth_sigma=0.35,taubin_iterations=2,upsample_factor=4)
    r=trimesh.load(str(d/f"{name}_SELECTED.stl")); r.merge_vertices(); r.update_faces(r.unique_faces())
    c=r.split(only_watertight=False)
    if c: r=max(c,key=lambda x:len(x.faces))
    trimesh.repair.fill_holes(r)
    if r.volume<0: r.invert()
    trimesh.repair.fix_normals(r); r.export(str(d/f"{name}_SELECTED.stl"))
    np.save(d/f"{name}_voxels{RES}.npy",vol.astype(np.uint8))
    man.append({"example":name,"selected_candidate":chosen["candidate"],"branch":BRANCH,"mean_iou":round(chosen["mean_iou"],4),
        "material_saved":round(chosen["material_saved_vs_hull"],4),"sf_fem":chosen["sf_fem"],"acs_safe":chosen["acs_certified_safe"],
        "fem_fallbacks":fb,"mesh_parts":len(r.split(only_watertight=False)),"watertight":bool(r.is_watertight),"sec":round(time.time()-t0)})
    print(f"{name}: sel={chosen['candidate']} IoU={chosen['mean_iou']:.3f} 省材={chosen['material_saved_vs_hull']*100:.0f}% FEM_SF={chosen['sf_fem']} ACS安全={chosen['acs_certified_safe']} 兜底={fb}")
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False)
print("DONE ->",OUT)
