import sys,numpy as np,torch,glob,random
sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from scipy.ndimage import binary_erosion,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,connectivity_preserving_carve
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
s6=generate_binary_structure(3,1);YIELD=50e6;lg=LoadGenerator();RES=48;B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3"
imgs=[b.split('/')[-1][:-4] for b in glob.glob(f"{B3}/*.png")];random.seed(99)  # 新种子=新三元组(与训练不重)
triples=[random.sample(imgs,3) for _ in range(30)]
dl=[];ml=[];vml=[];sfl=[];n=0
for ti,imn in enumerate(triples):
    try:
        gt,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(RES,RES),torch.device("cpu"),3,False)
        gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
        cal=calibrate_tfs_projection(gtn,search_res=96);hull,_=build_tfs_visual_hull(gtn,RES,calibration=cal)
        if hull.sum()<200: continue
        for v in [connectivity_preserving_carve(hull),hull&~binary_erosion(hull,structure=s6,iterations=2),hull&~binary_erosion(hull,structure=s6,iterations=1),hull]:
            if v.sum()<50: continue
            base=lg.standing_load(v.astype(np.float32))
            for M in (3.,6.,12.):
                f=base["force"]*M;o=voxel_fem_solve(v.astype(np.float32),f,base["fixed_mask"],verbose=False);vm=o["von_mises"][v>0.5]
                if vm.size==0 or np.isnan(vm).all(): continue
                vmax=float(np.nanmax(vm));dl.append(v.astype(np.uint8));ml.append(M);vml.append(vmax);sfl.append(YIELD/vmax);n+=1
    except: continue
    if (ti+1)%6==0: print(f"  {ti+1}/30 累计{n}",flush=True)
np.savez_compressed("/home/lin/xvxinyao/ShadowForge/data/scaleup_test.npz",dens=np.array(dl,np.uint8),mult=np.array(ml),vm=np.array(vml),sf=np.array(sfl))
sf=np.array(sfl);print(f"\n扩大测试集: {n}样本(新三元组) SF[{sf.min():.3f},{sf.max():.1f}] 不安全{int((sf<1.3).sum())}/安全{int((sf>=1.3).sum())}")
print("DONE")
