import sys,json,numpy as np,torch,importlib.util,glob
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from scipy.ndimage import binary_erosion,generate_binary_structure
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,connectivity_preserving_carve
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
dev=torch.device('cuda');YIELD=50e6;s6=generate_binary_structure(3,1);lg=LoadGenerator();RES=48;M=8.0
CP="/home/lin/xvxinyao/ShadowForge/checkpoints"
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
c=torch.load(f"{CP}/v40_geoattn/best.pt",map_location=dev,weights_only=False)
for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
hd=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));hd.load_state_dict(c["attn_scale_head_state_dict"]);p.eval()
def praw(d,f):
    x=build_x(d.astype(np.float32),f.astype(np.float32)).unsqueeze(0).to(dev)
    b,cc,nx,ny,nz=x.shape;g=p._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1)
    with torch.no_grad(): return float(p._scale_forward(torch.cat([x,g],1)).squeeze().cpu())
imgs=[basename.split('/')[-1][:-4] for basename in glob.glob("/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3/*.png")]
import random;random.seed(0)
triples=[random.sample(imgs,3) for _ in range(14)]
B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";data=[]
for ti,imn in enumerate(triples):
    try:
        gt,_=load_data_from_list([f"{B3}/{n}.png" for n in imn],True,(RES,RES),torch.device("cpu"),3,False)
        gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
        cal=calibrate_tfs_projection(gtn,search_res=96);hull,_=build_tfs_visual_hull(gtn,RES,calibration=cal)
        if hull.sum()<200: continue
        mem={"carve":connectivity_preserving_carve(hull),"hd2":hull&~binary_erosion(hull,structure=s6,iterations=2),
             "hd1":hull&~binary_erosion(hull,structure=s6,iterations=1),"hull":hull}
        for nm,v in mem.items():
            if v.sum()<50: continue
            L=lg.standing_load(v.astype(np.float32));f=L["force"]*M
            o=voxel_fem_solve(v.astype(np.float32),f,L["fixed_mask"],verbose=False);vm=o["von_mises"][v>0.5]
            if vm.size==0 or np.isnan(vm).all(): continue
            sf=YIELD/np.nanmax(vm);raw=praw(v.astype(np.float32),f)
            data.append((raw,sf,nm))
    except Exception as e: print(f"  triple{ti} skip:{str(e)[:30]}",flush=True);continue
    if (ti+1)%4==0: print(f"  triple {ti+1}/14, 累计{len(data)}样本",flush=True)
raw=np.array([d[0] for d in data]);sf=np.array([d[1] for d in data])
np.savez("/tmp/nullspace_femlabels.npz",raw=raw,sf=sf,names=[d[2] for d in data])
print(f"\n零空间FEM标注: {len(data)}样本 SF范围[{sf.min():.2f},{sf.max():.2f}] 不安全{int((sf<1.3).sum())}/安全{int((sf>=1.3).sum())}")
print(f"PINO raw vs log(SF) 相关(单调性): {np.corrcoef(raw,np.log10(np.maximum(sf,1e-6)))[0,1]:.3f}")
print("DONE")
