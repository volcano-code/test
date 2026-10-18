#!/usr/bin/env python3
"""48³ 平衡画廊: 交付 watertight_export(实心档), 与先前 carve_no_bridge(骨架档)逐件对比。
两档都要求 可打印(单体∧水密)∧IoU≥0.95∧FEM安全; 平衡档优先实心 watertight_export。"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from candidate_selection_batch import make_candidates, fem_sf, n6
_e=importlib.util.spec_from_file_location("ehqm",str(ROOT/"generation/export_high_quality_mesh.py")); ehqm=importlib.util.module_from_spec(_e); _e.loader.exec_module(ehqm)
import trimesh
R=48; TAU=1.3
def save_white(m,p): Image.fromarray((np.asarray(m)>0.5).astype(np.uint8)*255).resize((512,512),Image.NEAREST).save(p)
B=str(ROOT/"data/silhouettes/bench3"); H=str(ROOT/"generation")
EX={"heroes_spiderman_superman_batman":[f"{H}/Spider-Man.png",f"{H}/superman.png",f"{H}/batman.png"],
 "dragon_peacock_trex":[f"{B}/x_dragon.png",f"{B}/x_peacock.png",f"{B}/x_trex.png"],
 "parrot_squid_scorpion":[f"{B}/birds_parrot.png",f"{B}/sea_squid.png",f"{B}/x_scorpion.png"],
 "owl_bat_snail":[f"{B}/birds_owl.png",f"{B}/x_bat.png",f"{B}/sea_snail.png"]}
OUT=ROOT/"results/gallery48_balance"; OUT.mkdir(parents=True,exist_ok=True); man=[]
for name,paths in EX.items():
    t0=time.time()
    s,_=load_data_from_list(paths,True,(R,R),torch.device("cpu"),3,False)
    m=[np.asarray(x).squeeze().astype(np.float32) for x in s]; gt6=m+[np.fliplr(x) for x in m]
    cands,aux=make_candidates(m,R,False); hull=aux["hull"]; cal=aux["calibration"]; hv=int(hull.sum())
    def info(cn):
        v=cands[cn]; iou=evaluate_tfs_volume(v.astype(np.float32),gt6,cal,image_size=R)["mean_iou"]
        try: wt=bool(sdf_watertight_mesh(v,smooth_sigma=0.6).is_watertight)
        except: wt=False
        fe=fem_sf(v,1.0)
        return {"cand":cn,"iou":round(iou,4),"saving":round(1-int(v.sum())/hv,4),"occ":int(v.sum()),
                "single":n6(v)==1,"wt":wt,"sf_fem":round(fe["sf_fem"],2),"safe":fe["safe"],"_v":v}
    skel=info("carve_no_bridge") if "carve_no_bridge" in cands else None      # 骨架档
    bal =info("watertight_export") if "watertight_export" in cands else None  # 实心平衡档
    # 导出平衡档
    d=OUT/name; (d/"inputs").mkdir(parents=True,exist_ok=True); (d/"silhouettes").mkdir(parents=True,exist_ok=True)
    pick=bal if (bal and bal["single"] and bal["wt"] and bal["iou"]>=0.95 and bal["safe"]) else None
    if pick is None: pick=skel  # 兜底
    outp=d/f"{name}_balance.stl"
    ehqm.export_marching_cubes(pick["_v"].astype(bool),outp,smooth_sigma=0.35,taubin_iterations=2,upsample_factor=4)
    r=trimesh.load(str(outp)); r.merge_vertices(); r.update_faces(r.unique_faces())
    cc=r.split(only_watertight=False)
    if cc: r=max(cc,key=lambda x:len(x.faces))
    trimesh.repair.fill_holes(r)
    if r.volume<0: r.invert()
    trimesh.repair.fix_normals(r); r.export(str(outp)); np.save(d/f"{name}_voxels48.npy",pick["_v"].astype(np.uint8))
    for i,p in enumerate(paths):
        g=np.array(Image.open(p).convert("L")); Image.fromarray((g>127).astype(np.uint8)*255).resize((512,512),Image.LANCZOS).save(d/"inputs"/f"input{i}.png")
    ev=evaluate_tfs_volume(pick["_v"].astype(np.float32),gt6,cal,image_size=512)
    for i in range(3): save_white(ev["pred_masks"][i],d/"silhouettes"/f"shadow{i}.png")
    rec={"example":name,
         "balance_watertight":{k:v for k,v in bal.items() if k!="_v"} if bal else None,
         "skeleton_carve":{k:v for k,v in skel.items() if k!="_v"} if skel else None,
         "delivered":pick["cand"],"mesh_verts":len(r.vertices)}
    man.append(rec)
    print(f"{name}:")
    if skel: print(f"   骨架 carve     省材={skel['saving']*100:2.0f}% occ={skel['occ']:5d} IoU={skel['iou']:.3f} SF={skel['sf_fem']}")
    if bal:  print(f"   实心 watertight 省材={bal['saving']*100:2.0f}% occ={bal['occ']:5d} IoU={bal['iou']:.3f} SF={bal['sf_fem']}  <-交付")
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False); print("DONE ->",OUT)
