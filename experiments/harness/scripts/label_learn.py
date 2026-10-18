import sys, glob, random, time, json
from pathlib import Path
import numpy as np, torch
SF=Path("/home/lin/xvxinyao/ShadowForge")
for p in [SF, SF/"generation", SF/"experiments"]: sys.path.insert(0,str(p))
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume
from candidate_selection_batch import make_candidates, n6
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
YIELD=50e6; R=48; files=sorted(glob.glob(str(SF/"data/silhouettes/bench3/*.png"))); random.seed(3)
rows=[]; t0=time.time()
for it in range(40):
    tri=random.sample(files,3)
    try:
        s,_=load_data_from_list(tri,True,(R,R),torch.device('cpu'),3,False); m=[np.asarray(x).squeeze().astype(np.float32) for x in s]
        c,a=make_candidates(m,R,True); hv=int(a['hull'].sum()); gt6=m+[np.fliplr(x) for x in m]
    except Exception: continue
    for cn,v in c.items():
      try:
        v=v.astype(np.float32)
        if v.sum()<20: continue
        L=LoadGenerator().standing_load(v)
        fo=voxel_fem_solve(v,L['force'],L['fixed_mask'],verbose=False); vm=fo['von_mises'][v.astype(bool)]; vm=vm[np.isfinite(vm)]
        if vm.size==0: continue
        sfmax=YIELD/max(vm.max(),1e-9); sf99=YIELD/max(np.percentile(vm,99),1e-9)
        iou=evaluate_tfs_volume(v,gt6,a['calibration'],image_size=256)['mean_iou']
        rows.append(dict(sfmax=float(sfmax),sf99=float(sf99),ms=float(1-int(v.sum())/hv),occ=int(v.sum()),iou=float(iou),sb=int(n6(v.astype(bool))==1)))
      except Exception: continue
    if it%10==0: print(f'it{it} n={len(rows)} {time.time()-t0:.0f}s',flush=True)
json.dump(rows,open('/home/lin/xvxinyao/ShadowForge/experiments/harness/out/label_rows.json','w'))
print(f'n={len(rows)}',flush=True)
# === 可学习性: 同特征分别预测 log sfmax vs log sf99 ===
from sklearn.ensemble import GradientBoostingRegressor
X=np.array([[r['ms'],np.log10(max(r['occ'],1)),r['iou'],r['sb']] for r in rows])
def cv_mae(y,seeds=20):
    n=len(y); maes=[]
    for s in range(seeds):
        idx=np.random.default_rng(s).permutation(n); tr,te=idx[:3*n//4],idx[3*n//4:]
        gb=GradientBoostingRegressor(n_estimators=100,max_depth=3,learning_rate=0.05).fit(X[tr],y[tr])
        maes.append(np.mean(np.abs(gb.predict(X[te])-y[te])))
    return np.mean(maes)
ymax=np.log10([r['sfmax'] for r in rows]); y99=np.log10([r['sf99'] for r in rows])
print(f'log SFmax: std={ymax.std():.3f}  特征预测 CV-MAE={cv_mae(ymax):.3f}')
print(f'log SF99%: std={y99.std():.3f}  特征预测 CV-MAE={cv_mae(y99):.3f}')
ratio=np.median([rr['sf99']/rr['sfmax'] for rr in rows])
print('SF99/SFmax 比值 中位=%.1fx (确认SFmax被奇点压低)'%ratio)
print("DONE")
