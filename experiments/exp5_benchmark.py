import sys,json,numpy as np,torch,importlib.util
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from sklearn.isotonic import IsotonicRegression
from scipy.ndimage import binary_erosion,label,generate_binary_structure
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,connectivity_preserving_carve,evaluate_tfs_volume
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
dev=torch.device('cuda');YIELD=50e6;TAU=1.3;LOGTHR=np.log10(TAU);s6=generate_binary_structure(3,1);lg=LoadGenerator()
CP="/home/lin/xvxinyao/ShadowForge/checkpoints"
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py")
gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
def load_geoattn(ck):
    p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
    c=torch.load(ck,map_location=dev,weights_only=False)
    for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
    h=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));h.load_state_dict(c["attn_scale_head_state_dict"]);p.eval()
    for q in p.parameters(): q.requires_grad_(False)
    return p
pino=load_geoattn(f"{CP}/v40_geoattn/best.pt")
def praw(dens,force):
    x=build_x(dens.astype(np.float32),force.astype(np.float32)).unsqueeze(0).to(dev)
    b,c,nx,ny,nz=x.shape;g=pino._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1)
    with torch.no_grad(): return float(pino._scale_forward(torch.cat([x,g],1)).squeeze().cpu())
HARD=__import__('pathlib').Path(ROOT)/"results/shadowart_fem_hardset_v1"
dd=json.load(open("/home/lin/xvxinyao/ShadowForge/certification/pino_vs_fem_hard450.json"));paths=dd['paths'];fem=np.array(dd['fem_sf'])
idx=np.arange(len(paths));np.random.seed(0);np.random.shuffle(idx);ci=idx[:160]
mu=[];Y=[]
for j,i in enumerate(ci):
    d=np.load(HARD/paths[i],allow_pickle=True);mu.append(praw(d['density'],d['force']));Y.append(np.log10(max(fem[i],1e-6)))
mu=np.array(mu);Y=np.array(Y);h=len(mu)//2
iso=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(mu[:h],Y[:h]);Rr=np.maximum(iso.predict(mu[h:])-Y[h:],0);qm=np.quantile(Rr,0.95)
print(f"校准: raw[{mu.min():.2f},{mu.max():.2f}]->logSF[{iso.predict([mu.min()])[0]:.2f},{iso.predict([mu.max()])[0]:.2f}] q⁻={qm:.3f}",flush=True)
def accs(dens,force):
    m=praw(dens,force);return (iso.predict([m])[0]-qm>=LOGTHR), iso.predict([m])[0]
RES=48;B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";M=8.0  # 放大载荷使成员跨unsafe->safe且in-regime
def fem_sf(v,m):
    L=lg.standing_load(v.astype(np.float32));f=L["force"]*m
    o=voxel_fem_solve(v.astype(np.float32),f,L["fixed_mask"],verbose=False);vm=o["von_mises"][v>0.5]
    return (YIELD/np.nanmax(vm) if vm.size and not np.isnan(vm).all() else 0),f
rows=[]
for s,imn in {"A":["birds_owl","x_lobster","bugs_ant"],"C":["reptiles_turtle","x_scorpion","birds_eagle"]}.items():
    gt,_=load_data_from_list([f"{B3}/{n}.png" for n in imn],True,(RES,RES),torch.device("cpu"),3,False)
    gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
    cal=calibrate_tfs_projection(gtn,search_res=96);hull,_=build_tfs_visual_hull(gtn,RES,calibration=cal)
    mem={"雕除":connectivity_preserving_carve(hull),"镂空d2":hull&~binary_erosion(hull,structure=s6,iterations=2),
         "镂空d1":hull&~binary_erosion(hull,structure=s6,iterations=1),"外壳":hull}
    print(f"\n=== Set {s} 零空间硬基准 (载荷×{M:.0f}, 同剪影!) ===",flush=True)
    print(f"{'成员':>7}{'FEM-SF真值':>10}{'真安全':>7}{'ACCS判定':>9}{'正确':>6}",flush=True)
    for name,v in mem.items():
        sf,f=fem_sf(v,M);cert,logp=accs(v.astype(np.float32),f)
        truth=sf>=TAU;correct=(cert==truth);fp=(cert and not truth)
        rows.append((truth,cert))
        print(f"{name:>7}{sf:>10.2f}{str(truth):>7}{('认证安全' if cert else '拒/不安全'):>9}{('✓' if correct else ('误放行!' if fp else '✗保守')):>6}",flush=True)
TP=sum(c for t,c in rows if t);FP=sum(c for t,c in rows if not t);nsafe=sum(t for t,c in rows);nunsafe=len(rows)-nsafe
print(f"\n汇总: 总{len(rows)} 真安全{nsafe} 真不安全{nunsafe} | 误放行FP={FP}(应=0) 安全样本被认证{TP}/{nsafe}",flush=True)
print("DONE")
