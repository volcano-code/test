import sys,numpy as np,torch
sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from scipy.ndimage import binary_erosion,label,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,connectivity_preserving_carve,evaluate_tfs_volume
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
s6=generate_binary_structure(3,1);YIELD=50e6;TAU=1.3;lg=LoadGenerator();B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";R=48
def peak_sf(v):  # 物理屈服判据: 峰值von Mises
    L=lg.standing_load(v.astype(np.float32));o=voxel_fem_solve(v.astype(np.float32),L["force"],L["fixed_mask"],verbose=False)
    vm=o["von_mises"][v>0.5]
    return YIELD/np.nanmax(vm) if vm.size and not np.isnan(vm).all() else 0
for s,imn in {"A":["birds_owl","x_lobster","bugs_ant"],"C":["reptiles_turtle","x_scorpion","birds_eagle"]}.items():
    gt,_=load_data_from_list([f"{B3}/{n}.png" for n in imn],True,(R,R),torch.device("cpu"),3,False)
    gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
    cal=calibrate_tfs_projection(gtn,search_res=96);hull,_=build_tfs_visual_hull(gtn,R,calibration=cal);H=int(hull.sum())
    carve=connectivity_preserving_carve(hull)
    # 材料最小化序列: 外壳 → 镂空各深度 → 雕除(从大到小)
    seq=[("外壳",hull)]
    for d in (1,2,3): seq.append((f"镂空d{d}",hull&~binary_erosion(hull,structure=s6,iterations=d)))
    seq.append(("雕除",carve))
    seq=sorted(seq,key=lambda kv:-kv[1].sum())  # 材料从大到小
    print(f"\n=== Set {s} 认证材料最小化 (峰值SF, τ={TAU}) ===")
    print(f"{'成员':>8}{'材料%':>7}{'IoU':>7}{'峰值SF':>8}{'安全':>6}")
    safe_min=None
    for name,v in seq:
        mat=100*v.sum()/H;iou=evaluate_tfs_volume(v.astype(np.float32),gtn,cal,image_size=256)['mean_iou'];sf=peak_sf(v)
        ok=sf>=TAU
        if ok: safe_min=(name,mat,sf)  # 持续更新为更小的安全成员
        print(f"{name:>8}{mat:6.1f}%{iou:>7.3f}{sf:>8.2f}{'✓' if ok else '✗失效':>6}",flush=True)
    print(f"  → 最小安全落点(认证材料最小化目标): {safe_min[0] if safe_min else '无'} 材料{safe_min[1]:.1f}% SF{safe_min[2]:.2f}" if safe_min else "  → 全部失效",flush=True)
print("DONE")
