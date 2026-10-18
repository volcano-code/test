import sys,json,time,importlib.util
from pathlib import Path
import numpy as np,torch
ROOT=Path("/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"generation"));sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from run_all_baselines import evaluate_metrics, load_pino
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,repair_tfs_projection_gaps,evaluate_tfs_volume,resize_binary_volume
from structural.analysis.load_generator import LoadGenerator
from candidate_selection_batch import fit_acs, fem_sf, TAU
from scipy.ndimage import label,generate_binary_structure
_e=importlib.util.spec_from_file_location("ehqm",str(ROOT/"generation/export_high_quality_mesh.py"));ehqm=importlib.util.module_from_spec(_e);_e.loader.exec_module(ehqm)
import trimesh
s6=generate_binary_structure(3,1);dev=torch.device("cuda")
pino=load_pino(str(ROOT/"checkpoints/pino_v24_ufno_dual/best_model.pt"),dev)
iso,q_safe,q_unsafe=fit_acs(0.10);tau_log=float(np.log10(TAU));R=256
B=str(ROOT/"data/silhouettes/bench3");H=str(ROOT/"generation")
EX={"heroes":[f"{H}/Spider-Man.png",f"{H}/superman.png",f"{H}/batman.png"],
 "dragon_peacock_trex":[f"{B}/x_dragon.png",f"{B}/x_peacock.png",f"{B}/x_trex.png"],
 "frog_cricket_lobster":[f"{B}/reptiles_frog.png",f"{B}/bugs_cricket.png",f"{B}/x_lobster.png"],
 "parrot_squid_scorpion":[f"{B}/birds_parrot.png",f"{B}/sea_squid.png",f"{B}/x_scorpion.png"],
 "owl_bat_snail":[f"{B}/birds_owl.png",f"{B}/x_bat.png",f"{B}/sea_snail.png"]}
OUT=ROOT/"results/print5_gapfill256";OUT.mkdir(parents=True,exist_ok=True);man=[]
for name,paths in EX.items():
    t0=time.time()
    sils,_=load_data_from_list(paths,True,(R,R),torch.device("cpu"),3,False)
    masks=[np.asarray(x).squeeze().astype(np.float32) for x in sils]
    cal=calibrate_tfs_projection(masks,search_res=96,fix_axis=False)
    gap,_=repair_tfs_projection_gaps(masks,R,calibration=cal)   # gapfill外壳
    # 清微屑:保留最大6连通块
    lab,nc=label(gap,s6);sz=np.bincount(lab.ravel());sz[0]=0;big=sz.argmax();solid=(lab==big)
    iou=evaluate_tfs_volume(solid.astype(np.float32),masks,cal,image_size=R)["mean_iou"]
    # 认证(满外壳鲁棒安全):48下采样跑PINO+ACS + FEM
    v48=resize_binary_volume(solid,48);load=LoadGenerator().standing_load(v48.astype(np.float32))
    sf_pino=float(evaluate_metrics(v48.astype(np.float32),resize_binary_volume(gap,48).astype(np.float32),load["force"],pino,dev)["sf_min_pino"])
    pred=float(iso.predict([np.log10(max(sf_pino,1e-12))])[0]);acs_safe=bool(pred-q_safe>=tau_log);fe=fem_sf(v48,1.0)
    d=OUT/name;d.mkdir(exist_ok=True)
    ehqm.export_marching_cubes(solid.astype(bool),d/f"{name}.stl",smooth_sigma=0.35,taubin_iterations=2,upsample_factor=1)
    r=trimesh.load(str(d/f"{name}.stl"));r.merge_vertices();r.update_faces(r.unique_faces());c=r.split(only_watertight=False)
    if c:r=max(c,key=lambda x:len(x.faces))
    trimesh.repair.fill_holes(r)
    if r.volume<0:r.invert()
    trimesh.repair.fix_normals(r);r.export(str(d/f"{name}.stl"));np.save(d/f"{name}.npy",solid.astype(np.uint8))
    man.append({"example":name,"candidate":"gapfill_hull@256(清屑)","iou":round(iou,4),"saving":0.0,
        "acs_safe":acs_safe,"sf_pino_acs48":round(10**(pred-q_safe),2),"sf_fem48":fe["sf_fem"],"fem_safe":fe["safe"],
        "mesh_parts":len(r.split(only_watertight=False)),"watertight":bool(r.is_watertight),"sec":round(time.time()-t0)})
    print(f"{name}: IoU={iou:.3f} ACS安全={acs_safe}(SF≈{10**(pred-q_safe):.1f}) FEM_SF={fe['sf_fem']:.1f} 网格={len(r.split(only_watertight=False))}块 水密={r.is_watertight}")
json.dump(man,open(OUT/"MANIFEST.json","w"),indent=2,ensure_ascii=False);print("DONE")
