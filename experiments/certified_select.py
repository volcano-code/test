import sys,json,numpy as np,torch
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from sklearn.isotonic import IsotonicRegression
from scipy.ndimage import binary_erosion,distance_transform_edt,label,generate_binary_structure
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,connectivity_preserving_carve,evaluate_tfs_volume
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
dev='cuda';YIELD=50e6;TAU=1.3;LOGTHR=np.log10(TAU);s6=generate_binary_structure(3,1);lg=LoadGenerator()
ck="/home/lin/xvxinyao/ShadowForge/checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"
pino,_=load_surrogate(ck,torch.device(dev));pino.eval()
HARD=__import__('pathlib').Path(ROOT)/"results/shadowart_fem_hardset_v1"
def pino_logsf(dens,force):
    x=build_x(dens.astype(np.float32),force.astype(np.float32)).unsqueeze(0).to(dev)
    b,c,nx,ny,nz=x.shape;grid=pino._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1)
    with torch.no_grad(): raw=float(pino._scale_forward(torch.cat([x,grid],1)).squeeze().cpu())
    return raw  # raw ~ log10(SF)代理
# === 校准ACCS: base PINO 在 hard450 子集 ===
dd=json.load(open("/home/lin/xvxinyao/ShadowForge/certification/pino_vs_fem_hard450.json"))
paths=dd['paths'];fem=np.array(dd['fem_sf']);idx=np.arange(len(paths))
np.random.seed(0);np.random.shuffle(idx);cal_idx=idx[:160]
mu_cal=[];Y_cal=[]
for j,i in enumerate(cal_idx):
    d=np.load(HARD/paths[i],allow_pickle=True)
    mu_cal.append(pino_logsf(d['density'],d['force']));Y_cal.append(np.log10(max(fem[i],1e-6)))
    if (j+1)%40==0: print(f"  校准 {j+1}/160",flush=True)
mu_cal=np.array(mu_cal);Y_cal=np.array(Y_cal)
# 两折ACCS: iso(D_iso) + q_minus(D_conf)
h=len(mu_cal)//2
iso=IsotonicRegression(out_of_bounds='clip').fit(mu_cal[:h],Y_cal[:h])
R=iso.predict(mu_cal[h:])-Y_cal[h:];R=np.maximum(R,0)  # 过预测残差
qm=np.quantile(R,0.95)  # alpha=0.05下尾
def certify(mu): return iso.predict([mu])[0]-qm>=LOGTHR  # ACCS accept
print(f"ACCS校准完成: q⁻={qm:.3f}",flush=True)
# === 零空间成员 + 认证选择 ===
RES=48;B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3"
def fem_sf99(v):
    L=lg.standing_load(v.astype(np.float32));o=voxel_fem_solve(v.astype(np.float32),L["force"],L["fixed_mask"],verbose=False)
    vm=o["von_mises"][v>0.5];return (YIELD/np.nanpercentile(vm,99) if vm.size and not np.isnan(vm).all() else 0), L["force"]
for s,imn in {"A":["birds_owl","x_lobster","bugs_ant"],"C":["reptiles_turtle","x_scorpion","birds_eagle"]}.items():
    gt,_=load_data_from_list([f"{B3}/{n}.png" for n in imn],True,(RES,RES),torch.device("cpu"),3,False)
    gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
    cal=calibrate_tfs_projection(gtn,search_res=96);hull,_=build_tfs_visual_hull(gtn,RES,calibration=cal);H=int(hull.sum())
    members={"雕除(最小)":connectivity_preserving_carve(hull)}
    for d in (2,1): members[f"镂空d{d}"]=hull&~binary_erosion(hull,structure=s6,iterations=d)
    members["外壳(最大)"]=hull
    print(f"\n=== Set {s} 认证选择 (材料从小到大) ===")
    sel=None
    for name,v in members.items():
        L=lg.standing_load(v.astype(np.float32))
        mu=pino_logsf(v.astype(np.float32),L["force"])
        cert=certify(mu); sf99,_=fem_sf99(v)
        mat=100*v.sum()/H
        mark=""
        if cert and sel is None: sel=(name,mat,sf99);mark=" ← ACCS选中(最小且认证安全)"
        print(f"  {name:>10} 材料{mat:5.1f}% PINO-logSF={mu:.2f} ACCS认证={'安全' if cert else '拒'} FEM-SF99={sf99:.1f}{mark}",flush=True)
    if sel: print(f"  => 选中 {sel[0]}: 材料{sel[1]:.1f}% FEM-SF99={sel[2]:.1f} (真安全={sel[2]>=TAU})",flush=True)
print("DONE")
