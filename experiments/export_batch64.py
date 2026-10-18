#!/usr/bin/env python3
"""64³ 重走流程, 一批可打印件。每例: 完整7候选池打分 -> 选择 -> 导出 STL + 各方向剪影(512可辨认) + 输入图。
选择规则: 可打印(清尘后单体 ∧ 水密) ∧ IoU≥0.95 的候选里, 取材料最省(最轻); FEM SF 作安全审计。"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh, resize_binary_volume
from candidate_selection_batch import make_candidates, n6, keep_largest
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
_e=importlib.util.spec_from_file_location("ehqm",str(ROOT/"generation/export_high_quality_mesh.py")); ehqm=importlib.util.module_from_spec(_e); _e.loader.exec_module(ehqm)
import trimesh
YIELD=50e6; TAU=1.3; R=64; IOU_FLOOR=0.95
def fem_sf48(vol):  # 48单位约定: 下采样到48, cell=1.0 (PINO+ACS 工作档; 仅审计, 已知对高分辨率会假阳)
    v=keep_largest(resize_binary_volume(vol,48))
    if v.sum()<40: return None
    lg=LoadGenerator(cell_size=1.0); L=lg.standing_load(v.astype(np.float32))
    o=voxel_fem_solve(v.astype(np.float32),L["force"],L["fixed_mask"],cell_size=1.0,verbose=False)
    vm=o["von_mises"][v.astype(bool)]; return float(YIELD/np.nanmax(vm)) if vm.size else None
def save_white(m,p): Image.fromarray((np.asarray(m)>0.5).astype(np.uint8)*255).resize((512,512),Image.NEAREST).save(p)
B=str(ROOT/"data/silhouettes/bench3"); H=str(ROOT/"generation")
EX={"heroes_spiderman_superman_batman":[f"{H}/Spider-Man.png",f"{H}/superman.png",f"{H}/batman.png"],
 "dragon_peacock_trex":[f"{B}/x_dragon.png",f"{B}/x_peacock.png",f"{B}/x_trex.png"],
 "frog_cricket_lobster":[f"{B}/reptiles_frog.png",f"{B}/bugs_cricket.png",f"{B}/x_lobster.png"],
 "parrot_squid_scorpion":[f"{B}/birds_parrot.png",f"{B}/sea_squid.png",f"{B}/x_scorpion.png"],
 "owl_bat_snail":[f"{B}/birds_owl.png",f"{B}/x_bat.png",f"{B}/sea_snail.png"]}
OUT=ROOT/"results/batch64_print"; OUT.mkdir(parents=True,exist_ok=True); man=[]
for name,paths in EX.items():
    t0=time.time()
    s,_=load_data_from_list(paths,True,(R,R),torch.device("cpu"),3,False)
    m=[np.asarray(x).squeeze().astype(np.float32) for x in s]; gt6=m+[np.fliplr(x) for x in m]
    cands,aux=make_candidates(m,R,False); hull=aux["hull"]; cal=aux["calibration"]; hv=int(hull.sum())
    rows=[]
    for cn,vol in cands.items():
        iou=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=R)["mean_iou"]
        try: wt=bool(sdf_watertight_mesh(vol,smooth_sigma=0.6).is_watertight)
        except: wt=False
        rows.append({"candidate":cn,"iou":round(iou,4),"single_body":n6(vol)==1,"watertight":wt,
                     "saving":round(1-int(vol.sum())/max(hv,1),4),"occ":int(vol.sum()),"_v":vol})
    # 选择: 可打印 ∧ IoU>=0.95 里取最省材(最轻)
    printable=[r for r in rows if r["single_body"] and r["watertight"] and r["iou"]>=IOU_FLOOR]
    chosen=min(printable,key=lambda r:r["occ"]) if printable else None
    d=OUT/name; (d/"inputs").mkdir(parents=True,exist_ok=True); (d/"silhouettes").mkdir(parents=True,exist_ok=True)
    # 候选打分表落盘
    json.dump([{k:v for k,v in r.items() if k!="_v"} for r in rows],open(d/"candidate_scores.json","w"),indent=2,ensure_ascii=False)
    for i,p in enumerate(paths):
        g=np.array(Image.open(p).convert("L")); im=Image.fromarray((g>127).astype(np.uint8)*255).resize((512,512),Image.LANCZOS)
        im.save(d/"inputs"/f"input{i}.png")
    if chosen is None:
        man.append({"example":name,"status":"DEFER(无可打印≥0.95候选)","sec":round(time.time()-t0)}); print(f"{name}: DEFER"); continue
    vol=chosen["_v"]; outp=d/f"{name}_64.stl"
    ehqm.export_marching_cubes(vol.astype(bool),outp,smooth_sigma=0.35,taubin_iterations=2,upsample_factor=3)
    r=trimesh.load(str(outp)); r.merge_vertices(); r.update_faces(r.unique_faces())
    cc=r.split(only_watertight=False)
    if cc: r=max(cc,key=lambda x:len(x.faces))
    trimesh.repair.fill_holes(r)
    if r.volume<0: r.invert()
    trimesh.repair.fix_normals(r); r.export(str(outp)); np.save(d/f"{name}_voxels64.npy",vol.astype(np.uint8))
    ev=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=512)  # 模型投影剪影@512 可辨认
    for i in range(3): save_white(ev["pred_masks"][i],d/"silhouettes"/f"shadow{i}.png")
    sf=fem_sf48(vol)
    man.append({"example":name,"selected":chosen["candidate"],"iou_matched64":chosen["iou"],"saving":chosen["saving"],
        "single_body":chosen["single_body"],"watertight":chosen["watertight"],"fem_sf_48proxy_audit":round(sf,3) if sf else None,
        "fem_safe_48proxy":bool(sf and sf>=TAU),"mesh_verts":len(r.vertices),"mesh_parts":len(r.split(only_watertight=False)),"sec":round(time.time()-t0)})
    print(f"{name}: 选 {chosen['candidate']} IoU={chosen['iou']:.3f} 省材={chosen['saving']*100:.0f}% SF48代理={sf if sf is None else round(sf,2)} 网格={len(r.split(only_watertight=False))}块 {time.time()-t0:.0f}s")
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False); print("DONE ->",OUT)
