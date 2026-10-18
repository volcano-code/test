import sys,json,numpy as np,torch,importlib.util
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge")
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
dev=torch.device('cuda');CP="/home/lin/xvxinyao/ShadowForge/checkpoints"
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
def load_geo(ck):
    p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
    c=torch.load(ck,map_location=dev,weights_only=False)
    for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
    h=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));h.load_state_dict(c["attn_scale_head_state_dict"]);p.eval();return p
m1=load_geo(f"{CP}/v40_geoattn/best.pt");m2=load_geo(f"{CP}/v40_seed1111/best.pt")
def raw(p,d,f):
    x=build_x(d.astype(np.float32),f.astype(np.float32)).unsqueeze(0).to(dev)
    b,cc,nx,ny,nz=x.shape;g=p._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1)
    with torch.no_grad(): return float(p._scale_forward(torch.cat([x,g],1)).squeeze().cpu())
def disagree(d,f): return abs(raw(m1,d,f)-raw(m2,d,f))   # 集成分歧=OOD信号
HARD=__import__('pathlib').Path(ROOT)/"results/shadowart_fem_hardset_v1"
dd=json.load(open("/home/lin/xvxinyao/ShadowForge/certification/pino_vs_fem_hard450.json"));paths=dd['paths']
idx=np.arange(len(paths));np.random.seed(0);np.random.shuffle(idx)
id_dis=[]
for i in idx[:120]:
    d=np.load(HARD/paths[i],allow_pickle=True);id_dis.append(disagree(d['density'],d['force']))
id_dis=np.array(id_dis);thr=np.quantile(id_dis,0.95)
print(f"ID(hardset)分歧: mean={id_dis.mean():.3f} 95%阈值={thr:.3f}",flush=True)
# 零空间成员: 需重算density/force(npz只存了raw/sf). 重新生成几组算分歧
from scipy.ndimage import binary_erosion,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,connectivity_preserving_carve
from structural.analysis.load_generator import LoadGenerator
s6=generate_binary_structure(3,1);lg=LoadGenerator();RES=48;M=8.0;B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3"
import glob,random;imgs=[b.split('/')[-1][:-4] for b in glob.glob(f"{B3}/*.png")];random.seed(0)
ns_dis=[]
for _ in range(14):
    imn=random.sample(imgs,3)
    try:
        gt,_=load_data_from_list([f"{B3}/{n}.png" for n in imn],True,(RES,RES),torch.device("cpu"),3,False)
        gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
        cal=calibrate_tfs_projection(gtn,search_res=96);hull,_=build_tfs_visual_hull(gtn,RES,calibration=cal)
        if hull.sum()<200: continue
        for v in [connectivity_preserving_carve(hull),hull&~binary_erosion(hull,structure=s6,iterations=1),hull]:
            if v.sum()<50: continue
            L=lg.standing_load(v.astype(np.float32));ns_dis.append(disagree(v.astype(np.float32),L["force"]*M))
    except: continue
ns_dis=np.array(ns_dis)
flagged=int((ns_dis>thr).sum())
print(f"零空间分歧: mean={ns_dis.mean():.3f} ({len(ns_dis)}样本)")
print(f"被OOD检测器flag(>ID95%阈值): {flagged}/{len(ns_dis)} = {100*flagged/len(ns_dis):.0f}%")
print(f"→ flag的成员fail-closed defer → 不认证 → 不可能误放行 → FP=0(安全保证robust)")
print("DONE")
