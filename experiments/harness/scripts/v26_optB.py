import sys, importlib.util, glob, random, time
from pathlib import Path
import numpy as np, torch
SF_ROOT=Path("/home/lin/xvxinyao/ShadowForge"); TP_ROOT=Path("/data/xvxinyao/xvxinyao/total_project")
for p in [SF_ROOT, SF_ROOT/"generation", SF_ROOT/"experiments", TP_ROOT]: sys.path.insert(0,str(p))
import torch.nn.functional as F
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume
from candidate_selection_batch import make_candidates, n6
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
from structural.models.load_model import load_surrogate, forward_surrogate_physical
spec=importlib.util.spec_from_file_location("v26",TP_ROOT/"scripts/train_v26_shadowart_scale_sf.py")
v26m=importlib.util.module_from_spec(spec); spec.loader.exec_module(v26m)
dev=torch.device("cuda"); YIELD=50e6; TAU=1.3
# base (ShadowForge) + v26 (total_project)
base,_=load_surrogate(str(SF_ROOT/"checkpoints/pino_v24_ufno_dual/best_model.pt"),dev)
v26,_=load_surrogate(str(TP_ROOT/"checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"),dev); v26.enable_enhanced_scale_head()
ck=torch.load(str(TP_ROOT/"checkpoints/v26_shadowart_scale_sf/best_scale_sf.pt"),map_location=dev,weights_only=False)
v26.scale_head_v2.load_state_dict(ck["scale_head_v2_state_dict"])
if ck.get("scale_branch_state_dict"):
    for nm,st in ck["scale_branch_state_dict"].items(): getattr(v26,nm).load_state_dict(st)
base.eval(); v26.eval()
files=sorted(glob.glob(str(SF_ROOT/"data/silhouettes/bench3/*.png"))); random.seed(1); R=48
rows=[]; t0=time.time(); CT=['visual_hull','gap_filled_hull','carve_no_bridge','carve_thick2','carve_thick3','min_printable_bridge_r1','watertight_export']
for it in range(25):
    tri=random.sample(files,3)
    try:
        s,_=load_data_from_list(tri,True,(R,R),torch.device('cpu'),3,False); m=[np.asarray(x).squeeze().astype(np.float32) for x in s]
        c,a=make_candidates(m,R,True); hv=int(a['hull'].sum()); gt6=m+[np.fliplr(x) for x in m]
    except Exception: continue
    for cn,v in c.items():
      try:
        v=v.astype(np.float32); 
        if v.sum()<20: continue
        L=LoadGenerator().standing_load(v)
        fo=voxel_fem_solve(v,L['force'],L['fixed_mask'],verbose=False); vm=fo['von_mises'][v.astype(bool)]; vm=vm[np.isfinite(vm)]
        if vm.size==0: continue
        sf_fem=YIELD/max(vm.max(),1e-9)
        with torch.no_grad():
            rho=torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0).to(dev); ff=torch.from_numpy(L['force']).float().unsqueeze(0).to(dev)
            vmb=float((forward_surrogate_physical(base,rho,ff)['von_mises']*(rho>0.5).float()).max()); sf_b=YIELD/max(vmb,1e-9)
            x=v26m.build_x(v,L['force']); 
            if x.shape[-1]!=64: x=F.interpolate(x.unsqueeze(0),size=(64,)*3,mode='trilinear',align_corners=False).squeeze(0)
            plog=float(v26m.predict_log_vm(v26,x.unsqueeze(0).to(dev),train_branch=True).item()); sf_v=YIELD/max(10**plog,1e-9)
        iou=evaluate_tfs_volume(v,gt6,a['calibration'],image_size=256)['mean_iou']
        rows.append(dict(sf_fem=sf_fem,sf_base=sf_b,sf_v26=sf_v,ms=1-int(v.sum())/hv,occ=int(v.sum()),iou=iou,sb=int(n6(v.astype(bool))==1),wt=1,ct=CT.index(cn) if cn in CT else 0))
      except Exception as ex:
        print('  skip cand',cn,repr(ex)[:60],flush=True); continue
    if it%5==0: print(f'it{it} rows={len(rows)} {time.time()-t0:.0f}s',flush=True)
import json; json.dump([{k:float(v) for k,v in x.items()} for x in rows],open('/home/lin/xvxinyao/ShadowForge/experiments/harness/out/v26_optB_rows.json','w'))
print(f'生成候选 n={len(rows)}',flush=True)
# === 评测三方法 ===
from sklearn.ensemble import GradientBoostingRegressor
r=rows; n=len(r); tl=np.log10(TAU); ALPHA=0.10
femlog=np.array([np.log10(x['sf_fem']) for x in r])
def feats(key):
    return np.array([[np.log10(max(x[key],1e-6)),x['ms'],np.log10(max(x['occ'],1)),x['iou'],x['sb'],x['wt'],x['ct']] for x in r])
def eval_conformal(pred_sf_log):
    rng=np.random.default_rng(0); idx=rng.permutation(n); ca,ev=idx[:n//2],idx[n//2:]
    qs=np.quantile(pred_sf_log[ca]-femlog[ca],1-ALPHA); qu=np.quantile(femlog[ca]-pred_sf_log[ca],1-ALPHA)
    lcb=pred_sf_log[ev]-qs; ucb=pred_sf_log[ev]+qu; safe=lcb>=tl; unsafe=ucb<tl
    fem_ev=10**femlog[ev]
    return np.mean(safe|unsafe), (np.mean(safe&(fem_ev<TAU)) if safe.any() else 0.0)
def eval_optB(key):
    X=feats(key); rng=np.random.default_rng(0); idx=rng.permutation(n); tr,ca,ev=idx[:n//2],idx[n//2:3*n//4],idx[3*n//4:]
    gb=GradientBoostingRegressor(n_estimators=120,max_depth=3,learning_rate=0.05).fit(X[tr],femlog[tr])
    pred=gb.predict(X); resid=femlog[ca]-pred[ca]; q=np.quantile(resid,ALPHA); qu=np.quantile(-resid,ALPHA)
    lcb=pred[ev]+q; ucb=pred[ev]-qu  # q is lower-tail(neg): lcb=pred+q(<=0); ucb=pred-qu
    safe=lcb>=tl; unsafe=ucb<tl; fem_ev=10**femlog[ev]
    return np.mean(safe|unsafe),(np.mean(safe&(fem_ev<TAU)) if safe.any() else 0.0)
ab,fb=eval_conformal(np.array([np.log10(max(x['sf_base'],1e-6)) for x in r]))
av,fv=eval_conformal(np.array([np.log10(max(x['sf_v26'],1e-6)) for x in r]))
aB,fB=eval_optB('sf_v26')
print(f"真不安全占比={np.mean(10**femlog<TAU)*100:.0f}%")
print(f"base + conformal : 自主率={ab*100:.0f}% 假阳={fb*100:.1f}%")
print(f"v26  + conformal : 自主率={av*100:.0f}% 假阳={fv*100:.1f}%")
print(f"v26  + Option B  : 自主率={aB*100:.0f}% 假阳={fB*100:.1f}%")
print("DONE")
