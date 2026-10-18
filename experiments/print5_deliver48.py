#!/usr/bin/env python3
"""合法交付件: 48³ 权威管线(candidate_selection_batch 7候选 + dual_acs_fem_fallback + select)
此分辨率 ACS 在分布内, 管线真正替代 FEM(自主81%). 各导出选中件 STL(taubin) + 输入图512 + 剪影512."""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from run_all_baselines import evaluate_metrics, load_pino
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from structural.analysis.load_generator import LoadGenerator
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
def save_white(m,p): Image.fromarray((np.asarray(m)>0.5).astype(np.uint8)*255).resize((512,512),Image.NEAREST).save(p)
OUT=ROOT/"results/print5_deliver48"; OUT.mkdir(parents=True,exist_ok=True); man=[]
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
    d=OUT/name; (d/"inputs").mkdir(parents=True,exist_ok=True); (d/"silhouettes").mkdir(parents=True,exist_ok=True)
    if chosen is None:
        man.append({"example":name,"status":"DEFER","sec":round(time.time()-t0)}); print(f"{name}: DEFER"); continue
    vol=chosen["_vol"]
    outp=d/f"{name}_SELECTED.stl"
    ehqm.export_marching_cubes(vol.astype(bool),outp,smooth_sigma=0.35,taubin_iterations=2,upsample_factor=4)
    r=trimesh.load(str(outp)); r.merge_vertices(); r.update_faces(r.unique_faces())
    c=r.split(only_watertight=False)
    if c: r=max(c,key=lambda x:len(x.faces))
    trimesh.repair.fill_holes(r)
    if r.volume<0: r.invert()
    trimesh.repair.fix_normals(r); r.export(str(outp))
    np.save(d/f"{name}_voxels{RES}.npy",vol.astype(np.uint8))
    # 输入图(原始剪影,白底黑) + 模型剪影(管线标定投影), 含镜像, 512
    ev=evaluate_tfs_volume(vol.astype(np.float32),masks+[np.fliplr(m) for m in masks],cal,image_size=512)
    for i in range(3):
        save_white(ev["gt_masks"][i],d/"inputs"/f"input{i}.png"); save_white(ev["gt_masks"][i][:,::-1],d/"inputs"/f"input{i}_mirror.png")
        save_white(ev["pred_masks"][i],d/"silhouettes"/f"view{i}.png"); save_white(ev["pred_masks"][i+3],d/"silhouettes"/f"view{i}_mirror.png")
    fb_dec="ACS自主" if chosen["acs_certified_safe"] else "FEM兜底"
    man.append({"example":name,"selected_candidate":chosen["candidate"],"branch":BRANCH,"mean_iou":round(chosen["mean_iou"],4),
        "material_saved":round(chosen["material_saved_vs_hull"],4),"sf_fem":chosen["sf_fem"],"sf_pino":round(chosen["sf_pino"],3),
        "acs_safe":chosen["acs_certified_safe"],"certified_by":fb_dec,"fem_fallbacks_in_pool":fb,
        "mesh_parts":len(r.split(only_watertight=False)),"watertight":bool(r.is_watertight),"mesh_verts":len(r.vertices),"sec":round(time.time()-t0)})
    print(f"{name}: sel={chosen['candidate']} IoU={chosen['mean_iou']:.3f} 省材={chosen['material_saved_vs_hull']*100:.0f}% FEM_SF={chosen['sf_fem']} 认证={fb_dec} 网格={len(r.split(only_watertight=False))}块 {time.time()-t0:.0f}s")
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False)
print("DONE ->",OUT)
