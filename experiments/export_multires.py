#!/usr/bin/env python3
"""heroes 在 256/128/64/48 四档分辨率各导出最佳可打印件到一个文件夹。
可打印 = keep_largest(清尘)后 6连通=1 ∧ 水密;选 IoU@256 最高、并列取省材最高。
每档: STL(taubin) + 体素npy + 输入图(原始PNG512) + 模型剪影 + 指标(IoU/省材/可打印/FEM_SF[48,64档])。"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.ndimage import label, generate_binary_structure
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from candidate_selection_batch import make_candidates, n6
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
_e=importlib.util.spec_from_file_location("ehqm",str(ROOT/"generation/export_high_quality_mesh.py")); ehqm=importlib.util.module_from_spec(_e); _e.loader.exec_module(ehqm)
import trimesh
s6=generate_binary_structure(3,1); YIELD=50e6; TAU=1.3
def keep_largest(v):
    lab,nc=label(v,s6)
    if nc<=1: return v
    sz=np.bincount(lab.ravel()); sz[0]=0; return lab==sz.argmax()
def fem_sf(v,cell):
    lg=LoadGenerator(cell_size=cell); L=lg.standing_load(v.astype(np.float32))
    o=voxel_fem_solve(v.astype(np.float32),L["force"],L["fixed_mask"],cell_size=cell,verbose=False)
    vm=o["von_mises"][v.astype(bool)]; return float(YIELD/np.nanmax(vm)) if vm.size else None
def save_white(m,p): Image.fromarray((np.asarray(m)>0.5).astype(np.uint8)*255).resize((512,512),Image.NEAREST).save(p)

paths=[f"{ROOT}/generation/{x}.png" for x in ("Spider-Man","superman","batman")]
s,_=load_data_from_list(paths,True,(256,256),torch.device("cpu"),3,False)
m=[np.asarray(x).squeeze().astype(np.float32) for x in s]; gt6=m+[np.fliplr(x) for x in m]
OUT=ROOT/"results/heroes_multires_export"; OUT.mkdir(parents=True,exist_ok=True)
UPS={48:4,64:3,128:2,256:1}; man=[]
for R in (256,128,64,48):
    t0=time.time()
    cands,aux=make_candidates(m,R,False); hull=aux["hull"]; cal=aux["calibration"]; hv=int(hull.sum())
    rows=[]
    for cn,vol in cands.items():
        ck=keep_largest(vol)
        try: wt=bool(sdf_watertight_mesh(ck,smooth_sigma=0.6).is_watertight)
        except: wt=False
        if not (n6(ck)==1 and wt): continue
        iou=evaluate_tfs_volume(ck.astype(np.float32),gt6,cal,image_size=256)["mean_iou"]
        rows.append((cn,ck,iou,1-int(ck.sum())/max(hv,1)))
    if not rows: man.append({"resolution":R,"status":"无可打印候选"}); print(f"R={R}: 无可打印候选"); continue
    cn,vol,iou,sav=max(rows,key=lambda r:(r[2],r[3]))
    d=OUT/f"res{R}"; (d/"inputs").mkdir(parents=True,exist_ok=True); (d/"silhouettes").mkdir(parents=True,exist_ok=True)
    outp=d/f"heroes_res{R}.stl"
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
    sf=fem_sf(keep_largest(__import__("voxel_ops").resize_binary_volume(vol,48)),1.0) if R in (48,64,128) else None  # 48单位约定的48档代理SF(审计用)
    rec={"resolution":R,"selected_candidate":cn,"iou_at_256":round(iou,4),"material_saved":round(sav,4),
         "printable_single_body_watertight":True,"mesh_verts":len(r.vertices),"mesh_parts":len(r.split(only_watertight=False)),
         "fem_sf_48proxy_audit":round(sf,3) if sf else None,"sec":round(time.time()-t0)}
    man.append(rec); print(f"R={R}: {cn} IoU@256={iou:.4f} 省材={sav*100:.1f}% verts={len(r.vertices)} SF48代理={rec['fem_sf_48proxy_audit']} {time.time()-t0:.0f}s")
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False); print("DONE ->",OUT)
