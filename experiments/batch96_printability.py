#!/usr/bin/env python3
"""96³ 可打印性测试批次: 多个易识别例子, 交付 watertight(实心可打印件).
重点报可打印性(单体/水密/壁厚erosion/percentile-SF抗奇点), 出 STL + 各方向剪影@512 + 输入."""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
from candidate_selection_batch import make_candidates, n6
_e=importlib.util.spec_from_file_location("ehqm",str(ROOT/"generation/export_high_quality_mesh.py")); ehqm=importlib.util.module_from_spec(_e); _e.loader.exec_module(ehqm)
import trimesh
_S6=generate_binary_structure(3,1); YIELD=50e6; R=96
def ero(v): return float(binary_erosion(v,_S6).sum()/max(int(v.sum()),1))
def save_white(m,p): Image.fromarray((np.asarray(m)>0.5).astype(np.uint8)*255).resize((512,512),Image.NEAREST).save(p)
def sf_pct(v):  # 抗奇点 SF: 99分位 von Mises (cell=48/R 归一化)
    cell=48.0/R; L=LoadGenerator(cell_size=cell).standing_load(v.astype(np.float32))
    o=voxel_fem_solve(v.astype(np.float32),L["force"],L["fixed_mask"],cell_size=cell,verbose=False)
    vm=o["von_mises"][v.astype(bool)]; vm=vm[np.isfinite(vm)]
    return (float(YIELD/max(np.percentile(vm,99),1e-9)), float(YIELD/max(vm.max(),1e-9))) if vm.size else (None,None)
B=str(ROOT/"data/silhouettes/bench3")
EX={"lobster_bat_frog":["x_lobster","x_bat","reptiles_frog"],
    "owl_frog_dragon":["birds_owl","reptiles_frog","x_dragon"],
    "spider_dragon_owl":["x_spider","x_dragon","birds_owl"],
    "eagle_bee_frog":["birds_eagle","x_bee","reptiles_frog"],
    "turtle_peacock_bat":["reptiles_turtle","x_peacock","x_bat"]}
OUT=ROOT/"results/batch96_printability"; OUT.mkdir(parents=True,exist_ok=True); man=[]
for name,tri in EX.items():
    t0=time.time(); paths=[f"{B}/{n}.png" for n in tri]
    s,_=load_data_from_list(paths,True,(R,R),torch.device("cpu"),3,False); m=[np.asarray(x).squeeze().astype(np.float32) for x in s]
    cands,aux=make_candidates(m,R,True); hull=aux["hull"]; cal=aux["calibration"]; hv=int(hull.sum()); gt6=m+[np.fliplr(x) for x in m]
    v=cands["watertight_export"]
    iou=evaluate_tfs_volume(v.astype(np.float32),gt6,cal,image_size=256)["mean_iou"]
    try: wt=bool(sdf_watertight_mesh(v,smooth_sigma=0.6).is_watertight)
    except: wt=False
    sf99,sfmax=sf_pct(v)
    d=OUT/name; (d/"inputs").mkdir(parents=True,exist_ok=True); (d/"silhouettes").mkdir(parents=True,exist_ok=True)
    for i,n in enumerate(tri):
        g=np.array(Image.open(f"{B}/{n}.png").convert("L")); Image.fromarray((g>127).astype(np.uint8)*255).resize((512,512),Image.LANCZOS).save(d/"inputs"/f"input{i}_{n}.png")
    outp=d/f"{name}_96.stl"
    ehqm.export_marching_cubes(v.astype(bool),outp,smooth_sigma=0.5,taubin_iterations=3,upsample_factor=2)
    r=trimesh.load(str(outp)); r.merge_vertices(); r.update_faces(r.unique_faces()); cc=r.split(only_watertight=False)
    if cc: r=max(cc,key=lambda x:len(x.faces))
    trimesh.repair.fill_holes(r)
    if r.volume<0: r.invert()
    trimesh.repair.fix_normals(r); r.export(str(outp)); np.save(d/f"{name}_voxels96.npy",v.astype(np.uint8))
    ev=evaluate_tfs_volume(v.astype(np.float32),gt6,cal,image_size=512)
    for i in range(3): save_white(ev["pred_masks"][i],d/"silhouettes"/f"shadow{i}_{tri[i]}.png")
    rec={"example":name,"triple":tri,"resolution":96,"candidate":"watertight_export","mean_iou":round(iou,4),
        "material_saved":round(1-int(v.sum())/max(hv,1),4),"single_body":n6(v)==1,"watertight":wt,"erosion_survival":round(ero(v),3),
        "sf_fem_99pct":round(sf99,2) if sf99 else None,"sf_fem_max":round(sfmax,3) if sfmax else None,
        "mesh_verts":len(r.vertices),"mesh_parts":len(r.split(only_watertight=False)),"sec":round(time.time()-t0)}
    man.append(rec)
    print(f"{name}: IoU={iou:.3f} 省材={rec['material_saved']*100:.0f}% 单体={rec['single_body']} 水密={wt} ero={ero(v):.2f} SF99%={rec['sf_fem_99pct']} SFmax={rec['sf_fem_max']} {time.time()-t0:.0f}s",flush=True)
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False); print("DONE ->",OUT)
