import sys,numpy as np,torch,glob,random
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+"/generation")
from scipy.ndimage import generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,evaluate_tfs_volume,connectivity_preserving_carve
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
YIELD=50e6;lg=LoadGenerator();B3=ROOT+"/data/silhouettes/bench3";R=48
def sf(v,M=8.0):
    L=lg.standing_load(v.astype(np.float32));o=voxel_fem_solve(v.astype(np.float32),L["force"]*M,L["fixed_mask"],verbose=False)
    vm=o["von_mises"][v>0.5];return YIELD/np.nanmax(vm) if vm.size and not np.isnan(vm).all() else 0
imgs=[b.split('/')[-1][:-4] for b in glob.glob(f"{B3}/*.png")];random.seed(7);triples=[random.sample(imgs,3) for _ in range(24)]
iouH=[];iouC=[];matr=[];sfr=[]
for imn in triples:
    try:
        gt,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(R,R),torch.device("cpu"),3,False)
        gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
        cal=calibrate_tfs_projection(gtn,search_res=96);hull,_=build_tfs_visual_hull(gtn,R,calibration=cal)
        if hull.sum()<400: continue
        carve=connectivity_preserving_carve(hull)
        if carve.sum()<40: continue
        iouH.append(evaluate_tfs_volume(hull.astype(np.float32),gtn,cal,image_size=192)['mean_iou'])
        iouC.append(evaluate_tfs_volume(carve.astype(np.float32),gtn,cal,image_size=192)['mean_iou'])
        matr.append(hull.sum()/carve.sum())
        sh,sc=sf(hull),sf(carve)
        if sc>0 and sh>0: sfr.append(sh/sc)
    except Exception as e: continue
iouH,iouC,matr,sfr=map(np.array,(iouH,iouC,matr,sfr))
print(f"===== E1 验证: 阴影零空间欠定性 (N={len(matr)} 三元组, 48³, 同剪影) =====")
print(f"IoU 外壳={iouH.mean():.3f}±{iouH.std():.3f}  雕除={iouC.mean():.3f}±{iouC.std():.3f}  (恒定~0.99)")
print(f"材料跨度 外壳/雕除: 中位{np.median(matr):.1f}×  范围[{matr.min():.1f},{matr.max():.1f}]×  (论文 6-12×)")
print(f"SF跨度  外壳/雕除: 中位{np.median(sfr):.1f}×  最大{sfr.max():.1f}×  (论文 中位2.8×, 上限44.9×)")
print("DONE")
