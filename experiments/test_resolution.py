#!/usr/bin/env python3
"""测试指定体素分辨率下 heroes 各候选的 IoU(匹配口径+@256口径)、单连通、水密、省材。
用法: python experiments/test_resolution.py --resolution 128
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from candidate_selection_batch import make_candidates, n6

ap=argparse.ArgumentParser()
ap.add_argument("--resolution",type=int,required=True)
ap.add_argument("--example",default="heroes")
args=ap.parse_args()
R=args.resolution
H=str(ROOT/"generation"); B=str(ROOT/"data/silhouettes/bench3")
EX={"heroes":[f"{H}/Spider-Man.png",f"{H}/superman.png",f"{H}/batman.png"],
    "owl_bat_snail":[f"{B}/birds_owl.png",f"{B}/x_bat.png",f"{B}/sea_snail.png"],
    "parrot_squid_scorpion":[f"{B}/birds_parrot.png",f"{B}/sea_squid.png",f"{B}/x_scorpion.png"]}
paths=EX[args.example]
# 输入按 256 载入(真细节, REPRO 口径), 构建在 R, 镜像6视图
s,_=load_data_from_list(paths,True,(256,256),torch.device("cpu"),3,False)
m=[np.asarray(x).squeeze().astype(np.float32) for x in s]; gt6=m+[np.fliplr(x) for x in m]
cands,aux=make_candidates(m,R,False); hull=aux["hull"]; cal=aux["calibration"]; hv=int(hull.sum())
print(f"=== {args.example}  体素分辨率 R={R}  (输入载入256, 镜像6视图) ===")
print(f"{'候选':24s} {'IoU@R':>7s} {'IoU@256':>8s} {'单连通':>6s} {'水密':>5s} {'省材%':>6s}")
for cn,vol in cands.items():
    iR=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=R)["mean_iou"]
    i256=evaluate_tfs_volume(vol.astype(np.float32),gt6,cal,image_size=256)["mean_iou"]
    try: wt=bool(sdf_watertight_mesh(vol,smooth_sigma=0.6).is_watertight)
    except: wt=False
    sav=100*(1-int(vol.sum())/max(hv,1))
    print(f"{cn:24s} {iR:7.4f} {i256:8.4f} {str(n6(vol)==1):>6s} {str(wt):>5s} {sav:6.1f}")
print("可打印件 = 单连通=True ∧ 水密=True (watertight_export / min_printable_bridge_r1)")
