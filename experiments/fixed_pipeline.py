import sys,json,numpy as np,torch,importlib.util,glob,random
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from sklearn.isotonic import IsotonicRegression
from scipy.ndimage import binary_erosion,generate_binary_structure
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,connectivity_preserving_carve
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
dev=torch.device('cuda');YIELD=50e6;TAU=1.3;LOGTHR=np.log10(TAU);s6=generate_binary_structure(3,1);lg=LoadGenerator();RES=48;M=8.0
CP="/home/lin/xvxinyao/ShadowForge/checkpoints";B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3"
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
c=torch.load("/home/lin/xvxinyao/ShadowForge/data/v40_finetuned.pt",map_location=dev,weights_only=False)
for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
h=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));h.load_state_dict(c["attn_scale_head_state_dict"]);p.eval()
def raw_of(d,f):
    x=build_x(d.astype(np.float32),f.astype(np.float32)).unsqueeze(0).to(dev)
    b,cc,nx,ny,nz=x.shape;g=p._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1)
    with torch.no_grad(): return float(p._scale_forward(torch.cat([x,g],1)).squeeze().cpu())
# 校准(hardset+free增广,同B3)
HARD=__import__('pathlib').Path(ROOT)/"results/shadowart_fem_hardset_v1"
dd=json.load(open("/home/lin/xvxinyao/ShadowForge/certification/pino_vs_fem_hard450.json"));paths=dd['paths'];fem=np.array(dd['fem_sf'])
ix=np.arange(len(paths));np.random.seed(0);np.random.shuffle(ix)
cr=[];cy=[]
for i in ix[:120]:
    d=np.load(HARD/paths[i],allow_pickle=True);cr.append(raw_of(d['density'],d['force']));cy.append(np.log10(max(fem[i],1e-6)))
ag=np.load("/home/lin/xvxinyao/ShadowForge/data/aug.npz");te=set(np.load("/home/lin/xvxinyao/ShadowForge/data/test_idx.npy").tolist())
for j in [k for k in range(len(ag['dens'])) if k not in te][:200]:
    v=ag['dens'][j].astype(np.float32);f=lg.standing_load(v)["force"]*ag['mult'][j];cr.append(raw_of(v,f));cy.append(np.log10(max(ag['sf'][j],1e-6)))
cr=np.array(cr);cy=np.array(cy);o=np.argsort(cr);crs,cys=cr[o],cy[o];hh=len(crs)//2
iso=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(crs[:hh],cys[:hh])
resid=np.maximum(iso.predict(crs[hh:])-cys[hh:],0);qm=np.quantile(resid,0.80)  # α=0.2
# === FIXED分支零空间成员 ===
imgs=[b.split('/')[-1][:-4] for b in glob.glob(f"{B3}/*.png")];random.seed(55);trip=[random.sample(imgs,3) for _ in range(12)]
raws=[];Ys=[]
for imn in trip:
    try:
        gt,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(RES,RES),torch.device("cpu"),3,False)
        gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
        cal=calibrate_tfs_projection(gtn,search_res=96,fix_axis=True)  # FIXED!
        hull,_=build_tfs_visual_hull(gtn,RES,calibration=cal)
        if hull.sum()<200: continue
        for v in [connectivity_preserving_carve(hull),hull&~binary_erosion(hull,structure=s6,iterations=1),hull]:
            if v.sum()<50: continue
            L=lg.standing_load(v.astype(np.float32));f=L["force"]*M
            ov=voxel_fem_solve(v.astype(np.float32),f,L["fixed_mask"],verbose=False);vm=ov["von_mises"][v>0.5]
            if vm.size==0 or np.isnan(vm).all(): continue
            raws.append(raw_of(v.astype(np.float32),f));Ys.append(np.log10(max(YIELD/np.nanmax(vm),1e-6)))
    except: continue
raws=np.array(raws);Ys=np.array(Ys);truth=Ys>=LOGTHR
acc=iso.predict(raws)-qm>=LOGTHR
fp=int((acc&~truth).sum());tp=int((acc&truth).sum());nsafe=int(truth.sum())
print(f"FIXED分支零空间: {len(raws)}样本 安全{nsafe}/不安全{len(raws)-nsafe}")
print(f"微调PINO在FIXED上相关={np.corrcoef(raws,Ys)[0,1]:.3f}")
print(f"ACCS(α=0.2): FP={fp}(应0) 认证率={tp}/{nsafe}={tp/max(nsafe,1):.0%}")
print("DONE")
