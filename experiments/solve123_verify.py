import sys,os,numpy as np,torch,glob,random,json
os.environ.setdefault("OMP_NUM_THREADS","4")
sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/scripts")
from scipy.ndimage import label,generate_binary_structure
from scipy.stats import spearmanr
import trimesh
from datasets import load_data_from_list
from voxel_ops import (calibrate_tfs_projection,build_tfs_visual_hull,connectivity_preserving_carve,
    minimize_material_printable,make_watertight_printable,sdf_watertight_mesh,evaluate_tfs_volume)
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve

s6=generate_binary_structure(3,1);YIELD=50e6;TAU=1.3;lg=LoadGenerator();B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3"
def n6(v):_,n=label(v,structure=s6);return n
def sf_at(v,M):
    L=lg.standing_load(v.astype(np.float32))
    o=voxel_fem_solve(v.astype(np.float32),L["force"]*M,L["fixed_mask"],verbose=False)
    vm=o["von_mises"][v>0.5]
    if vm.size==0 or np.isnan(vm).all(): return None
    mx=float(np.nanmax(vm))
    return None if mx<=0 else YIELD/mx

imgs=[b.split('/')[-1][:-4] for b in glob.glob(f"{B3}/*.png")]
random.seed(2026);triples=[random.sample(imgs,3) for _ in range(20)]   # 扩到70,取有效≥50
R=128;RF=48;RC=64
sav=[];iou=[];sb=[];wt=[]                       # ② N>=50 规模化
min_sf={1.0:[],3.0:[],6.0:[]}                   # ① 最小体在不同载荷下的SF
conv48=[];conv64=[]                             # ③ 收敛:最小体SF @48 vs @64

for ti,imn in enumerate(triples):
  try:
    # ---- ② 128³ 规模化刻画 ----
    gt,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(R,R),torch.device("cpu"),3,False)
    gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
    cal=calibrate_tfs_projection(gtn,search_res=96);hull,_=build_tfs_visual_hull(gtn,R,calibration=cal)
    if int(hull.sum())<800: continue
    v,_=minimize_material_printable(hull,bridge_radius=1);vw,info=make_watertight_printable(v,hull)
    m=sdf_watertight_mesh(vw,smooth_sigma=0.6)
    sav.append(info['material_saved_ratio']*100)
    iou.append(evaluate_tfs_volume(vw.astype(np.float32),gtn,cal,image_size=256)['mean_iou'])
    sb.append(1 if n6(vw)==1 else 0);wt.append(1 if m.is_watertight else 0)

    # ---- ① 48³ 最小可打印体 在 M=1(自重)/3/6 下的 SF ----
    gl,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(RF,RF),torch.device("cpu"),3,False)
    gln=[np.asarray(x).squeeze().astype(np.float32) for x in gl]
    cl=calibrate_tfs_projection(gln,search_res=96);hl,_=build_tfs_visual_hull(gln,RF,calibration=cl)
    mv,_=minimize_material_printable(hl,bridge_radius=1)
    if mv.sum()>=50:
        for M in (1.0,3.0,6.0):
            s=sf_at(mv,M)
            if s is not None: min_sf[M].append(s)
        # ---- ③ 收敛:同一最小体定义在 48³ vs 64³ 的 SF(用于排序稳定) ----
        if len(conv48)<10:
            s48=sf_at(mv,1.0)
            gc,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(RC,RC),torch.device("cpu"),3,False)
            gcn=[np.asarray(x).squeeze().astype(np.float32) for x in gc]
            cc=calibrate_tfs_projection(gcn,search_res=96);hc,_=build_tfs_visual_hull(gcn,RC,calibration=cc)
            mc,_=minimize_material_printable(hc,bridge_radius=1)
            s64=sf_at(mc,1.0) if mc.sum()>=50 else None
            if s48 is not None and s64 is not None: conv48.append(s48);conv64.append(s64)
  except Exception as e:
    print(f"  triple{ti} skip: {type(e).__name__} {e}",flush=True);continue
  if (ti+1)%7==0: print(f"  {ti+1}/70 有效{len(sav)} 收敛对{len(conv48)}",flush=True)

def ms(x):x=np.array(x);return f"{x.mean():.3f}±{x.std():.3f}"
out={}
print(f"\n===== ② 规模化 (N={len(sav)} 三元组, free, 128³) =====")
print(f"省材%: {ms(sav)} 范围[{min(sav):.0f},{max(sav):.0f}]")
print(f"IoU:   {ms(iou)}  单体率{100*np.mean(sb):.0f}%  水密率{100*np.mean(wt):.0f}%")
out['N']=len(sav);out['saving_mean']=float(np.mean(sav));out['saving_std']=float(np.std(sav))
out['iou_mean']=float(np.mean(iou));out['single_body_pct']=float(100*np.mean(sb));out['watertight_pct']=float(100*np.mean(wt))

print(f"\n===== ① 最小可打印体的认证内生需求 (48³) =====")
out['binding']={}
for M in (1.0,3.0,6.0):
    a=np.array(min_sf[M]);unsafe=int((a<TAU).sum())
    print(f"载荷×{M:.0f}: 最小体 SF[{a.min():.2f},{a.max():.2f}] 中位{np.median(a):.2f}  不安全(SF<{TAU}): {unsafe}/{len(a)} = {100*unsafe/len(a):.0f}%")
    out['binding'][f'M{int(M)}']={'n':len(a),'unsafe':unsafe,'unsafe_pct':round(100*unsafe/len(a),1),'sf_median':round(float(np.median(a)),2)}

print(f"\n===== ③ 分辨率收敛 (最小体 SF @48³ vs @64³, n={len(conv48)}) =====")
if len(conv48)>=4:
    rho,p=spearmanr(conv48,conv64)
    print(f"Spearman 排序相关 = {rho:.3f} (p={p:.1e})  →排序稳定" if rho>0.7 else f"Spearman={rho:.3f}")
    out['convergence']={'n':len(conv48),'spearman':round(float(rho),3),'sf48':[round(x,2) for x in conv48],'sf64':[round(x,2) for x in conv64]}
else:
    print("收敛对不足")
    out['convergence']={'n':len(conv48)}

json.dump(out,open("/home/lin/xvxinyao/ShadowForge/data/solve123_VERIFY.json","w"),indent=2)
print("\nSAVED -> results/solve123_results.json\nDONE")
