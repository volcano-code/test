import sys,numpy as np,torch,glob,random,itertools
sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from scipy.ndimage import binary_erosion,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,connectivity_preserving_carve
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
s6=generate_binary_structure(3,1);YIELD=50e6;lg=LoadGenerator();RES=48;B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3"
imgs=[b.split('/')[-1][:-4] for b in glob.glob(f"{B3}/*.png")];random.seed(7)
triples=[random.sample(imgs,3) for _ in range(40)]
dens_l=[];mult_l=[];vm_l=[];sf_l=[]
n=0
for ti,imn in enumerate(triples):
    try:
        gt,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(RES,RES),torch.device("cpu"),3,False)
        gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
        cal=calibrate_tfs_projection(gtn,search_res=96);hull,_=build_tfs_visual_hull(gtn,RES,calibration=cal)
        if hull.sum()<200: continue
        mem=[connectivity_preserving_carve(hull),hull&~binary_erosion(hull,structure=s6,iterations=2),
             hull&~binary_erosion(hull,structure=s6,iterations=1),hull]
        for v in mem:
            if v.sum()<50: continue
            base=lg.standing_load(v.astype(np.float32))
            for M in (2.,4.,8.,16.):
                f=base["force"]*M;o=voxel_fem_solve(v.astype(np.float32),f,base["fixed_mask"],verbose=False)
                vm=o["von_mises"][v>0.5]
                if vm.size==0 or np.isnan(vm).all(): continue
                vmax=float(np.nanmax(vm))*1.0
                dens_l.append(v.astype(np.uint8));mult_l.append(M);vm_l.append(vmax);sf_l.append(YIELD/vmax);n+=1
    except Exception as e: continue
    if (ti+1)%8==0: print(f"  triple {ti+1}/40, 累计{n}样本",flush=True)
np.savez_compressed("/home/lin/xvxinyao/ShadowForge/data/aug.npz",dens=np.array(dens_l,dtype=np.uint8),mult=np.array(mult_l),vm=np.array(vm_l),sf=np.array(sf_l))
sf=np.array(sf_l)
print(f"\n增广数据: {n}样本 SF[{sf.min():.3f},{sf.max():.2f}] 不安全{int((sf<1.3).sum())}/安全{int((sf>=1.3).sum())}")
print("DONE")
