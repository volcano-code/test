#!/usr/bin/env python3
"""跳过soft: 几何候选(含gapfill外壳)@48³ → PINO+ACS+FEM认证 → 选【最高IoU】的可打印安全件 → 256 taubin导出.
认证在48³原生(PINO+ACS有效,无下采样). 省材消融后补."""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, torch
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from run_all_baselines import evaluate_metrics, load_pino
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from structural.analysis.load_generator import LoadGenerator
from candidate_selection_batch import make_candidates, fit_acs, fem_sf, n6, TAU
_e=importlib.util.spec_from_file_location("ehqm",str(ROOT/"generation/export_high_quality_mesh.py")); ehqm=importlib.util.module_from_spec(_e); _e.loader.exec_module(ehqm)
import trimesh
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
pino=load_pino(str(ROOT/"checkpoints/pino_v24_ufno_dual/best_model.pt"),dev)
iso,q_safe,q_unsafe=fit_acs(0.10); tau_log=float(np.log10(TAU)); RES=48
B=str(ROOT/"data/silhouettes/bench3"); H=str(ROOT/"generation")
EX={"heroes_spiderman_superman_batman":[f"{H}/Spider-Man.png",f"{H}/superman.png",f"{H}/batman.png"],
 "dragon_peacock_trex":[f"{B}/x_dragon.png",f"{B}/x_peacock.png",f"{B}/x_trex.png"],
 "frog_cricket_lobster":[f"{B}/reptiles_frog.png",f"{B}/bugs_cricket.png",f"{B}/x_lobster.png"],
 "parrot_squid_scorpion":[f"{B}/birds_parrot.png",f"{B}/sea_squid.png",f"{B}/x_scorpion.png"],
 "owl_bat_snail":[f"{B}/birds_owl.png",f"{B}/x_bat.png",f"{B}/sea_snail.png"]}
OUT=ROOT/"results/print5_highiou"; OUT.mkdir(parents=True,exist_ok=True); man=[]
for name,paths in EX.items():
    t0=time.time()
    sils,_=load_data_from_list(paths,True,(RES,RES),torch.device("cpu"),3,False)
    masks=[np.asarray(x).squeeze().astype(np.float32) for x in sils]
    cands,aux=make_candidates(masks,RES,False); hull=aux["hull"]; cal=aux["calibration"]
    rows=[]
    for cn,vol in cands.items():
        iou=evaluate_tfs_volume(vol.astype(np.float32),masks,cal,image_size=RES)["mean_iou"]
        try: wt=bool(sdf_watertight_mesh(vol,smooth_sigma=0.6).is_watertight)
        except: wt=False
        load=LoadGenerator().standing_load(vol.astype(np.float32))
        sf_pino=float(evaluate_metrics(vol.astype(np.float32),hull.astype(np.float32),load["force"],pino,dev)["sf_min_pino"])
        pred=float(iso.predict([np.log10(max(sf_pino,1e-12))])[0]); fe=fem_sf(vol,1.0)
        rows.append(dict(candidate=cn,iou=iou,single6=n6(vol)==1,watertight=wt,saving=1-int(vol.sum())/max(int(hull.sum()),1),
            acs_safe=bool(pred-q_safe>=tau_log),acs_unsafe=bool(pred+q_unsafe<tau_log),fem_safe=fe["safe"],sf_fem=fe["sf_fem"],_vol=vol))
    # 可打印 ∧ (ACS安全 或 FEM兜底安全) → 选最高IoU
    ok=[r for r in rows if r["single6"] and r["watertight"] and (r["acs_safe"] or (not r["acs_unsafe"] and r["fem_safe"]))]
    chosen=max(ok,key=lambda r:r["iou"]) if ok else None
    d=OUT/name; d.mkdir(exist_ok=True)
    if chosen is None:
        man.append({"example":name,"status":"DEFER"}); print(f"{name}: DEFER"); continue
    vol=chosen["_vol"]
    ehqm.export_marching_cubes(vol.astype(bool),d/f"{name}_SELECTED.stl",smooth_sigma=0.35,taubin_iterations=2,upsample_factor=4)
    r=trimesh.load(str(d/f"{name}_SELECTED.stl")); r.merge_vertices(); r.update_faces(r.unique_faces())
    c=r.split(only_watertight=False)
    if c: r=max(c,key=lambda x:len(x.faces))
    trimesh.repair.fill_holes(r)
    if r.volume<0: r.invert()
    trimesh.repair.fix_normals(r); r.export(str(d/f"{name}_SELECTED.stl")); np.save(d/f"{name}_voxels{RES}.npy",vol.astype(np.uint8))
    man.append({"example":name,"selected":chosen["candidate"],"iou":round(chosen["iou"],4),"saving":round(chosen["saving"],4),
        "sf_fem":chosen["sf_fem"],"acs_safe":chosen["acs_safe"],"mesh_parts":len(r.split(only_watertight=False)),"watertight":bool(r.is_watertight),"sec":round(time.time()-t0)})
    print(f"{name}: sel={chosen['candidate']} IoU={chosen['iou']:.3f} 省材={chosen['saving']*100:.0f}% FEM_SF={chosen['sf_fem']} ACS安全={chosen['acs_safe']} 网格={len(r.split(only_watertight=False))}块")
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False); print("DONE ->",OUT)
