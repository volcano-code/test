import sys,numpy as np,torch
sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from scipy.ndimage import binary_erosion,label,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,evaluate_tfs_volume,connectivity_preserving_carve
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
s6=generate_binary_structure(3,1);YIELD=50e6;lg=LoadGenerator();B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";R=64
SETS={"A":["birds_owl","x_lobster","bugs_ant"],"C":["reptiles_turtle","x_scorpion","birds_eagle"]}
def fem_sf(v):
    L=lg.standing_load(v.astype(np.float32));o=voxel_fem_solve(v.astype(np.float32),L["force"],L["fixed_mask"],verbose=False)
    vm=o["von_mises"][v>0.5]
    if vm.size==0 or np.isnan(vm).all(): return 0,0
    return YIELD/np.nanmax(vm), YIELD/np.nanpercentile(vm,99)   # 峰值, 99分位
for s,imn in SETS.items():
    gt,_=load_data_from_list([f"{B3}/{n}.png" for n in imn],True,(R,R),torch.device("cpu"),3,False)
    gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
    cal=calibrate_tfs_projection(gtn,search_res=min(96,R));hull,_=build_tfs_visual_hull(gtn,R,calibration=cal);H=int(hull.sum())
    # 零空间样本: 外壳 -> 镂空(腐蚀壳,保剪影近似) -> 雕除(最小)
    carve=connectivity_preserving_carve(hull)
    pts={"外壳(最大)":hull}
    for d in (1,2):
        shell=hull&~binary_erosion(hull,structure=s6,iterations=d)
        pts[f"镂空d{d}"]=shell
    pts["雕除(最小)"]=carve
    print(f"\n=== Set {s} 剪影零空间 (hull={H}, 64³) ===")
    print(f"{'零空间成员':>14}{'材料%':>7}{'IoU':>7}{'SF峰值':>8}{'SF@99':>8}{'6连通':>6}")
    for name,v in pts.items():
        iou=evaluate_tfs_volume(v.astype(np.float32),gtn,cal,image_size=256)['mean_iou']
        mx,p99=fem_sf(v);_,n6=label(v,structure=s6)
        print(f"{name:>14}{100*v.sum()/H:6.1f}%{iou:>7.3f}{mx:>8.1f}{p99:>8.1f}{n6:>6}",flush=True)
print("DONE")
