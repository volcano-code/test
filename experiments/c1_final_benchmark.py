#!/usr/bin/env python3
"""C1: 用 THE FINAL 选择器(select_final, free∪fixed+壁厚门槛+均衡省材) 重做选择基准, 取代过时 Table4。
报告: 选择率, 交付候选类型分布, 均衡省材, IoU, FEM安全率, ACS自主率; 并列旧 dual_acs_fem_fallback 对比。"""
import sys, json, random, time
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import binary_erosion, generate_binary_structure
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from run_all_baselines import evaluate_metrics, load_pino
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from structural.analysis.load_generator import LoadGenerator
from candidate_selection_batch import make_candidates, fit_acs, fem_sf, n6, select_final, select, TAU
_S6=generate_binary_structure(3,1)
def ero(v): return float(binary_erosion(v,_S6).sum()/max(int(v.sum()),1))
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
pino=load_pino(str(ROOT/"checkpoints/pino_v24_ufno_dual/best_model.pt"),dev)
iso,q_safe,q_unsafe=fit_acs(0.10); tau_log=float(np.log10(TAU)); R=48
sil=ROOT/"data/silhouettes/bench3"; names=sorted(p.stem for p in sil.glob("*.png"))
rng=random.Random(20260624); N=24; triples=[rng.sample(names,3) for _ in range(N)]
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
            "sf_pino":sfp,"raw_pino_safe":sfp>=TAU,"acs_certified_safe":bool(pred-q_safe>=tau_log),
            "acs_certified_unsafe":bool(pred+q_unsafe<tau_log),"fem_safe":fe["safe"],"sf_fem":fe["sf_fem"]})
    return out
fin=[]; old=[]; types={}
for i,tr in enumerate(triples):
    paths=[str(sil/f"{n}.png") for n in tr]
    s,_=load_data_from_list(paths,True,(R,R),torch.device("cpu"),3,False); m=[np.asarray(x).squeeze().astype(np.float32) for x in s]
    pool=rows_for(m,False,"free")+rows_for(m,True,"fixed")
    ch,info=select_final(pool)
    if ch is not None:
        fin.append({"saving":ch["material_saved_vs_hull"],"iou":ch["mean_iou"],"safe":ch["fem_safe"],
                    "auto":info["decided_by"]=="ACS自主","cand":ch["candidate"],"branch":ch["branch"]})
        types[ch["candidate"]]=types.get(ch["candidate"],0)+1
    # 旧策略(单 free 分支, dual_acs_fem_fallback) 对比
    rows_free=[r for r in pool if r["branch"]=="free"]
    co,fb=select(rows_free,"dual_acs_fem_fallback")
    if co is not None: old.append({"saving":co["material_saved_vs_hull"],"safe":co["fem_safe"],"cand":co["candidate"]})
    print(f"[{i+1}/{N}] {tr} -> final={ch['candidate'] if ch else 'DEFER'}",flush=True)
def agg(L,k): return float(np.mean([x[k] for x in L])) if L else None
res={"n_cases":N,
 "select_final":{"selection_rate":len(fin)/N,"mean_saving":agg(fin,"saving"),"mean_iou":agg(fin,"iou"),
   "fem_safe_rate":agg(fin,"safe"),"acs_autonomy":agg(fin,"auto"),"delivered_types":types},
 "old_dual_acs_fem_fallback_free":{"selection_rate":len(old)/N,"mean_saving":agg(old,"saving"),"fem_safe_rate":agg(old,"safe")}}
json.dump(res,open(ROOT/"results/C1_selectfinal_benchmark.json","w"),indent=2,ensure_ascii=False)
print(json.dumps(res,indent=2,ensure_ascii=False))
