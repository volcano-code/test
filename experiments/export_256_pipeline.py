#!/usr/bin/env python3
"""256³ 整体管线(几何部分): 候选生成+选择+导出。
诚实标注: PINO@256 OOM / FEM@256 不可行 -> 结构认证在部署分辨率(48/128)做, 此处只跑几何+可打印性。
交付: STL(taubin) + 体素 + 各方向剪影@512 + 输入 + 全候选打分。"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from candidate_selection_batch import make_candidates, n6, keep_largest
_e=importlib.util.spec_from_file_location("ehqm",str(ROOT/"generation/export_high_quality_mesh.py")); ehqm=importlib.util.module_from_spec(_e); _e.loader.exec_module(ehqm)
import trimesh
_S6=generate_binary_structure(3,1)
def ero(v): return float(binary_erosion(v,_S6).sum()/max(int(v.sum()),1))
def save_white(m,p): Image.fromarray((np.asarray(m)>0.5).astype(np.uint8)*255).resize((512,512),Image.NEAREST).save(p)
R=256; B=str(ROOT/"data/silhouettes/bench3")
EX={"lobster_bat_frog":["x_lobster","x_bat","reptiles_frog"],
    "owl_frog_dragon":["birds_owl","reptiles_frog","x_dragon"]}
OUT=ROOT/"results/gallery_256_pipeline"; OUT.mkdir(parents=True,exist_ok=True); man=[]
for name,tri in EX.items():
    t0=time.time(); paths=[f"{B}/{n}.png" for n in tri]
    s,_=load_data_from_list(paths,True,(R,R),torch.device("cpu"),3,False); m=[np.asarray(x).squeeze().astype(np.float32) for x in s]
    cands,aux=make_candidates(m,R,True); hull=aux["hull"]; cal=aux["calibration"]; hv=int(hull.sum()); gt6=m+[np.fliplr(x) for x in m]
    rows=[]
    for cn,vol in cands.items():
        iou=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=R)["mean_iou"]
        try: wt=bool(sdf_watertight_mesh(vol,smooth_sigma=0.6).is_watertight)
        except: wt=False
        rows.append({"candidate":cn,"mean_iou":round(iou,4),"single_body":n6(vol)==1,"watertight":wt,
            "material_saved":round(1-int(vol.sum())/max(hv,1),4),"erosion_survival":round(ero(vol),3),"occ":int(vol.sum()),"_v":vol})
    # 选择(几何部分: 单体∧水密∧IoU≥0.95∧壁厚≥0.15 里取 IoU 最高;省材并列取省材)
    ok=[r for r in rows if r["single_body"] and r["watertight"] and r["mean_iou"]>=0.95 and r["erosion_survival"]>=0.15]
    chosen=max(ok,key=lambda r:(r["mean_iou"],r["material_saved"])) if ok else max(rows,key=lambda r:r["mean_iou"])
    d=OUT/name; (d/"inputs").mkdir(parents=True,exist_ok=True); (d/"silhouettes").mkdir(parents=True,exist_ok=True)
    json.dump([{k:v for k,v in r.items() if k!="_v"} for r in rows],open(d/"candidate_scores.json","w"),indent=2,ensure_ascii=False)
    for i,n in enumerate(tri):
        g=np.array(Image.open(f"{B}/{n}.png").convert("L")); Image.fromarray((g>127).astype(np.uint8)*255).resize((512,512),Image.LANCZOS).save(d/"inputs"/f"input{i}_{n}.png")
    vol=chosen["_v"]; outp=d/f"{name}_256.stl"
    ehqm.export_marching_cubes(vol.astype(bool),outp,smooth_sigma=0.6,taubin_iterations=3,upsample_factor=1)
    r=trimesh.load(str(outp)); r.merge_vertices(); r.update_faces(r.unique_faces()); cc=r.split(only_watertight=False)
    if cc: r=max(cc,key=lambda x:len(x.faces))
    trimesh.repair.fill_holes(r)
    if r.volume<0: r.invert()
    trimesh.repair.fix_normals(r); r.export(str(outp)); np.save(d/f"{name}_voxels256.npy",vol.astype(np.uint8))
    ev=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=512)
    for i in range(3): save_white(ev["pred_masks"][i],d/"silhouettes"/f"shadow{i}_{tri[i]}.png")
    rec={"example":name,"triple":tri,"resolution":256,"selected":chosen["candidate"],"mean_iou":chosen["mean_iou"],
        "material_saved":chosen["material_saved"],"single_body":chosen["single_body"],"watertight":chosen["watertight"],
        "erosion_survival":chosen["erosion_survival"],"mesh_verts":len(r.vertices),"mesh_parts":len(r.split(only_watertight=False)),
        "certification":"N/A@256 (PINO OOM/FEM@256 infeasible; certify at deployment res 48/128)","sec":round(time.time()-t0)}
    man.append(rec); print(f"{name}: {chosen['candidate']} IoU={chosen['mean_iou']:.3f} 省材={chosen['material_saved']*100:.0f}% 单体={chosen['single_body']} 水密={chosen['watertight']} verts={len(r.vertices)} {time.time()-t0:.0f}s")
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False); print("DONE ->",OUT)
