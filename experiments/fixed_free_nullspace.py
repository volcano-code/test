import sys,numpy as np,torch
sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from scipy.ndimage import label,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,connectivity_preserving_carve,minimize_material_printable,evaluate_tfs_volume,analyze_connected_components
s6=generate_binary_structure(3,1);B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";R=128
def n6(v):_,n=label(v,structure=s6);return n
for s,imn in {"A":["birds_owl","x_lobster","bugs_ant"],"C":["reptiles_turtle","x_scorpion","birds_eagle"]}.items():
    gt,_=load_data_from_list([f"{B3}/{n}.png" for n in imn],True,(R,R),torch.device("cpu"),3,False)
    gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
    print(f"\n=== Set {s} 两分支零空间选择 (128³) ===")
    print(f"{'分支':>6}{'外壳材料':>9}{'最小成员材料':>12}{'零空间跨度':>11}{'最小IoU':>8}{'6连通':>6}")
    for branch,fix in [("fixed",True),("free",False)]:
        cal=calibrate_tfs_projection(gtn,search_res=96,fix_axis=fix)
        hull,_=build_tfs_visual_hull(gtn,R,calibration=cal);H=int(hull.sum())
        # 选最优=可打印最小成员
        v,info=minimize_material_printable(hull,bridge_radius=1)
        iou=evaluate_tfs_volume(v.astype(np.float32),gtn,cal,image_size=256)['mean_iou']
        print(f"{branch:>6}{H:>9}{int(v.sum()):>12}{H/max(int(v.sum()),1):>10.1f}x{iou:>8.4f}{n6(v):>6}",flush=True)
print("DONE")
