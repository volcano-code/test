#!/usr/bin/env python3
"""heroes 在 256/128/64/48 各导出【未做体积最小化】的件 = gap_filled_hull(清尘后单体)。
实心、保真最高、省材≈0。每档: STL(taubin) + 体素 + 输入图(原始PNG512) + 模型剪影 + 指标。"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from candidate_selection_batch import make_candidates, n6
_e=importlib.util.spec_from_file_location("ehqm",str(ROOT/"generation/export_high_quality_mesh.py")); ehqm=importlib.util.module_from_spec(_e); _e.loader.exec_module(ehqm)
import trimesh
def save_white(m,p): Image.fromarray((np.asarray(m)>0.5).astype(np.uint8)*255).resize((512,512),Image.NEAREST).save(p)
paths=[f"{ROOT}/generation/{x}.png" for x in ("Spider-Man","superman","batman")]
s,_=load_data_from_list(paths,True,(256,256),torch.device("cpu"),3,False)
m=[np.asarray(x).squeeze().astype(np.float32) for x in s]; gt6=m+[np.fliplr(x) for x in m]
OUT=ROOT/"results/heroes_multires_NOMIN"; OUT.mkdir(parents=True,exist_ok=True)
UPS={48:4,64:3,128:2,256:1}; man=[]
for R in (256,128,64,48):
    t0=time.time()
    cands,aux=make_candidates(m,R,False); hull=aux["hull"]; cal=aux["calibration"]; hv=int(hull.sum())
    vol=cands["gap_filled_hull"]  # 未做体积最小化
    iou=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=256)["mean_iou"]
    try: wt=bool(sdf_watertight_mesh(vol,smooth_sigma=0.6).is_watertight)
    except: wt=False
    d=OUT/f"res{R}"; (d/"inputs").mkdir(parents=True,exist_ok=True); (d/"silhouettes").mkdir(parents=True,exist_ok=True)
    outp=d/f"heroes_res{R}_nomin.stl"
    ehqm.export_marching_cubes(vol.astype(bool),outp,smooth_sigma=0.35,taubin_iterations=2,upsample_factor=UPS[R])
    r=trimesh.load(str(outp)); r.merge_vertices(); r.update_faces(r.unique_faces())
    cc=r.split(only_watertight=False)
    if cc: r=max(cc,key=lambda x:len(x.faces))
    trimesh.repair.fill_holes(r)
    if r.volume<0: r.invert()
    trimesh.repair.fix_normals(r); r.export(str(outp))
    np.save(d/f"heroes_res{R}_voxels.npy",vol.astype(np.uint8))
    for i,p in enumerate(paths):
        g=np.array(Image.open(p).convert("L")); im=Image.fromarray((g>127).astype(np.uint8)*255).resize((512,512),Image.LANCZOS)
        im.save(d/"inputs"/f"input{i}.png"); im.transpose(Image.FLIP_LEFT_RIGHT).save(d/"inputs"/f"input{i}_mirror.png")
    ev=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=512)
    for i in range(3):
        save_white(ev["pred_masks"][i],d/"silhouettes"/f"view{i}.png"); save_white(ev["pred_masks"][i+3],d/"silhouettes"/f"view{i}_mirror.png")
    rec={"resolution":R,"candidate":"gap_filled_hull(未体积最小化)","iou_at_256":round(iou,4),
         "material_saved":round(1-int(vol.sum())/max(hv,1),4),"single_body":n6(vol)==1,"watertight":wt,
         "occupied_voxels":int(vol.sum()),"mesh_verts":len(r.vertices),"mesh_parts":len(r.split(only_watertight=False)),"sec":round(time.time()-t0)}
    man.append(rec); print(f"R={R}: gapfill IoU@256={iou:.4f} 省材={rec['material_saved']*100:.1f}% 单体={rec['single_body']} 水密={wt} occ={int(vol.sum())} verts={len(r.vertices)} {time.time()-t0:.0f}s")
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False); print("DONE ->",OUT)
