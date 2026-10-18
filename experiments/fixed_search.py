#!/usr/bin/env python3
"""搜索 select_final 会选中 fixed 的案例(供实物展示)。扫 N 三元组, free∪fixed 池 -> select_final,
收集 chosen.branch=='fixed' 的例子, 报 IoU/省材/SF/自主。"""
import sys, json, random
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import binary_erosion, generate_binary_structure
ROOT=Path("/home/lin/xvxinyao/ShadowForge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"generation")); sys.path.insert(0,str(ROOT/"experiments"))
from datasets import load_data_from_list
from run_all_baselines import evaluate_metrics, load_pino
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from structural.analysis.load_generator import LoadGenerator
from candidate_selection_batch import make_candidates, fit_acs, fem_sf, n6, select_final, TAU
_S6=generate_binary_structure(3,1)
def ero(v): return float(binary_erosion(v,_S6).sum()/max(int(v.sum()),1))
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
pino=load_pino(str(ROOT/"checkpoints/pino_v24_ufno_dual/best_model.pt"),dev)
iso,q_safe,q_unsafe=fit_acs(0.10); tau_log=float(np.log10(TAU)); R=48
sil=ROOT/"data/silhouettes/bench3"; names=sorted(p.stem for p in sil.glob("*.png"))
rng=random.Random(99); N=40; triples=[rng.sample(names,3) for _ in range(N)]
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
            "sf_pino":sfp,"acs_certified_safe":bool(pred-q_safe>=tau_log),"acs_certified_unsafe":bool(pred+q_unsafe<tau_log),
            "fem_safe":fe["safe"],"sf_fem":fe["sf_fem"]})
    return out
fixed_cases=[]; nfix=nfree=ndefer=0
for i,tr in enumerate(triples):
    paths=[str(sil/f"{n}.png") for n in tr]
    s,_=load_data_from_list(paths,True,(R,R),torch.device("cpu"),3,False); m=[np.asarray(x).squeeze().astype(np.float32) for x in s]
    pool=rows_for(m,False,"free")+rows_for(m,True,"fixed")
    ch,info=select_final(pool)
    if ch is None: ndefer+=1; br="DEFER"
    elif ch["branch"]=="fixed":
        nfix+=1; br="fixed"
        fixed_cases.append({"triple":tr,"candidate":ch["candidate"],"iou":round(ch["mean_iou"],4),
            "saving":round(ch["material_saved_vs_hull"],4),"sf_fem":round(ch["sf_fem"],2),"decided_by":info["decided_by"]})
    else: nfree+=1; br="free"
    print(f"[{i+1}/{N}] {tr} -> {br}"+(f" {ch['candidate']} IoU={ch['mean_iou']:.3f} 省材={ch['material_saved_vs_hull']*100:.0f}% SF={ch['sf_fem']:.1f}" if ch else ""),flush=True)
print(f"\n汇总: fixed选中={nfix} free选中={nfree} DEFER={ndefer} (共{N})")
print("=== FIXED 被选中案例(供实物展示)===")
for c in fixed_cases: print(f"  {'+'.join(c['triple']):40s} {c['candidate']:16s} IoU={c['iou']:.3f} 省材={c['saving']*100:.0f}% SF={c['sf_fem']} {c['decided_by']}")
json.dump(fixed_cases,open(ROOT/"results/fixed_friendly_cases.json","w"),indent=2,ensure_ascii=False)
print("DONE -> results/fixed_friendly_cases.json")
