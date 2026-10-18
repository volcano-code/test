#!/usr/bin/env python3
"""导出最敦实(最不易拉丝)的 3 个 fixed 案例: select_final 正式选中件 + STL + 各方向剪影@512 + 输入。"""
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
B=str(ROOT/"data/silhouettes/bench3")
TOP3={"turtle_peacock_bat":["reptiles_turtle","x_peacock","x_bat"],
      "owl_turtle_peacock":["birds_owl","reptiles_turtle","x_peacock"],
      "peacock_parrot_ladybug":["x_peacock","birds_parrot","x_ladybug"]}
def rows_for(masks,fix_axis,branch):
    cands,aux=make_candidates(masks,R,fix_axis); hull=aux["hull"]; cal=aux["calibration"]; hv=int(hull.sum()); gt6=masks+[np.fliplr(x) for x in masks]; out=[]
    for cn,vol in cands.items():
        iou=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=R)["mean_iou"]
        try: wt=bool(sdf_watertight_mesh(vol,smooth_sigma=0.6).is_watertight)
        except: wt=False
        load=LoadGenerator().standing_load(vol.astype(np.float32))
        sfp=float(evaluate_metrics(vol.astype(np.float32),hull.astype(np.float32),load["force"],pino,dev)["sf_min_pino"])
        pred=float(iso.predict([np.log10(max(sfp,1e-12))])[0]); fe=fem_sf(vol,1.0)
        out.append({"candidate":cn,"branch":branch,"mean_iou":iou,"component_count_6":n6(vol),"watertight":wt,
            "material_saved_vs_hull":1-int(vol.sum())/max(hv,1),"occupied_voxels":int(vol.sum()),"erosion_survival":ero(vol),
            "sf_pino":sfp,"acs_certified_safe":bool(pred-q_safe>=tau_log),"acs_certified_unsafe":bool(pred+q_unsafe<tau_log),
            "fem_safe":fe["safe"],"sf_fem":fe["sf_fem"],"_vol":vol,"_cal":cal})
    return out
OUT=ROOT/"results/gallery_fixed_top3"; OUT.mkdir(parents=True,exist_ok=True); man=[]
for name,tri in TOP3.items():
    t0=time.time(); paths=[f"{B}/{n}.png" for n in tri]
    s,_=load_data_from_list(paths,True,(R,R),torch.device("cpu"),3,False); m=[np.asarray(x).squeeze().astype(np.float32) for x in s]
    pool=rows_for(m,False,"free")+rows_for(m,True,"fixed")
    ch,info=select_final(pool)
    d=OUT/name; (d/"inputs").mkdir(parents=True,exist_ok=True); (d/"silhouettes").mkdir(parents=True,exist_ok=True)
    for i,p in enumerate(paths):
        g=np.array(Image.open(p).convert("L")); Image.fromarray((g>127).astype(np.uint8)*255).resize((512,512),Image.LANCZOS).save(d/"inputs"/f"input{i}.png")
    if ch is None: man.append({"example":name,"status":"DEFER"}); print(f"{name}: DEFER"); continue
    vol=ch["_vol"]; cal=ch["_cal"]; outp=d/f"{name}_{ch['branch']}.stl"
    ehqm.export_marching_cubes(vol.astype(bool),outp,smooth_sigma=0.35,taubin_iterations=2,upsample_factor=4)
    r=trimesh.load(str(outp)); r.merge_vertices(); r.update_faces(r.unique_faces())
    cc=r.split(only_watertight=False)
    if cc: r=max(cc,key=lambda x:len(x.faces))
    trimesh.repair.fill_holes(r)
    if r.volume<0: r.invert()
    trimesh.repair.fix_normals(r); r.export(str(outp)); np.save(d/f"{name}_voxels48.npy",vol.astype(np.uint8))
    ev=evaluate_tfs_volume(vol.astype(np.float32),m+[np.fliplr(x) for x in m],cal,image_size=512)
    for i in range(3): save_white(ev["pred_masks"][i],d/"silhouettes"/f"shadow{i}.png")
    man.append({"example":name,"triple":tri,"branch":ch["branch"],"selected":ch["candidate"],"decided_by":info["decided_by"],
        "mean_iou":round(ch["mean_iou"],4),"material_saved":round(ch["material_saved_vs_hull"],4),"sf_fem":round(ch["sf_fem"],2),
        "erosion_survival":round(ch["erosion_survival"],3),"watertight":bool(r.is_watertight),"mesh_parts":len(r.split(only_watertight=False))})
    print(f"{name}: {ch['branch']}/{ch['candidate']} IoU={ch['mean_iou']:.3f} 省材={ch['material_saved_vs_hull']*100:.0f}% SF={ch['sf_fem']:.1f} 认证={info['decided_by']} 水密={r.is_watertight} {time.time()-t0:.0f}s")
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False); print("DONE ->",OUT)
