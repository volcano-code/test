import sys,numpy as np,torch,glob,random
sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from scipy.ndimage import label,generate_binary_structure
import trimesh
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,connectivity_preserving_carve,minimize_material_printable,make_watertight_printable,sdf_watertight_mesh,evaluate_tfs_volume
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
s6=generate_binary_structure(3,1);YIELD=50e6;lg=LoadGenerator();B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";R=128;RF=48
def n6(v):_,n=label(v,structure=s6);return n
def sf_peak(v):
    vv=v;L=lg.standing_load(vv.astype(np.float32));o=voxel_fem_solve(vv.astype(np.float32),L["force"]*8,L["fixed_mask"],verbose=False);vm=o["von_mises"][vv>0.5]
    return YIELD/np.nanmax(vm) if vm.size and not np.isnan(vm).all() else 0
imgs=[b.split('/')[-1][:-4] for b in glob.glob(f"{B3}/*.png")];random.seed(2026);triples=[random.sample(imgs,3) for _ in range(24)]
sav=[];iou=[];sb=[];wt=[];sfr=[]
for ti,imn in enumerate(triples):
    try:
        gt,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(R,R),torch.device("cpu"),3,False)
        gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
        cal=calibrate_tfs_projection(gtn,search_res=96);hull,_=build_tfs_visual_hull(gtn,R,calibration=cal);H=int(hull.sum())
        if H<800: continue
        v,_=minimize_material_printable(hull,bridge_radius=1);vw,info=make_watertight_printable(v,hull)
        m=sdf_watertight_mesh(vw,smooth_sigma=0.6)
        sav.append(info['material_saved_ratio']*100);iou.append(evaluate_tfs_volume(vw.astype(np.float32),gtn,cal,image_size=256)['mean_iou'])
        sb.append(1 if n6(vw)==1 else 0);wt.append(1 if m.is_watertight else 0)
        # SF跨度: 48³ hull vs carve
        gtl,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(RF,RF),torch.device("cpu"),3,False)
        gl=[np.asarray(x).squeeze().astype(np.float32) for x in gtl];cl=calibrate_tfs_projection(gl,search_res=96);hl,_=build_tfs_visual_hull(gl,RF,calibration=cl)
        cv=connectivity_preserving_carve(hl);sh=sf_peak(hl);sc=sf_peak(cv)
        if sc>0: sfr.append(sh/sc)
    except Exception as e: continue
    if (ti+1)%6==0: print(f"  {ti+1}/24 有效{len(sav)}",flush=True)
import numpy as np
def ms(x):x=np.array(x);return f"{x.mean():.3f}±{x.std():.3f}"
print(f"\n=== 规模化(N={len(sav)}三元组,free分支,128³) ===")
print(f"省材%:     {ms(sav)}  范围[{min(sav):.0f},{max(sav):.0f}]")
print(f"IoU:       {ms(iou)}")
print(f"单体率(6连通=1): {100*np.mean(sb):.0f}%")
print(f"水密率:    {100*np.mean(wt):.0f}%")
print(f"SF跨度(外壳/雕除): 中位{np.median(sfr):.1f}× 范围[{min(sfr):.1f},{max(sfr):.1f}] (同剪影!)")
print("DONE")
