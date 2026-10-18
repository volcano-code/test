import sys,json,numpy as np,torch,importlib.util,time
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from structural.analysis.load_generator import LoadGenerator
dev=torch.device('cuda');CP="/home/lin/xvxinyao/ShadowForge/checkpoints";lg=LoadGenerator()
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
c=torch.load(f"{CP}/v40_geoattn/best.pt",map_location=dev,weights_only=False)
SB=["scale_lifting","scale_down1","scale_enc2","scale_down2","scale_bottleneck"]
for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
head=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));head.load_state_dict(c["attn_scale_head_state_dict"])
# 冻结全部, 解冻 scale_branch + attn头
for q in p.parameters(): q.requires_grad_(False)
train_params=[]
for nm in SB:
    for q in getattr(p,nm).parameters(): q.requires_grad_(True);train_params.append(q)
for q in head.parameters(): q.requires_grad_(True);train_params.append(q)
def xg_of(dens,force):
    x=build_x(dens.astype(np.float32),force.astype(np.float32)).unsqueeze(0).to(dev)
    b,cc,nx,ny,nz=x.shape;g=p._get_grid((nx,ny,nz),x.device).expand(b,-1,-1,-1,-1);return torch.cat([x,g],1)
# 数据: hardset子集 + 零空间增广(留出测试)
HARD=__import__('pathlib').Path(ROOT)/"results/shadowart_fem_hardset_v1"
dd=json.load(open("/home/lin/xvxinyao/ShadowForge/certification/pino_vs_fem_hard450.json"));paths=dd['paths']
idx=np.arange(len(paths));np.random.seed(0);np.random.shuffle(idx)
hs=[]
for i in idx[:200]:
    d=np.load(HARD/paths[i],allow_pickle=True);vm=float(np.array(d['vm_max']))
    hs.append((d['density'].astype(np.float32),d['force'].astype(np.float32),np.log10(max(vm,1e-3))))
aug=np.load("/home/lin/xvxinyao/ShadowForge/data/aug.npz");A_d,A_m,A_vm=aug['dens'],aug['mult'],aug['vm']
perm=np.random.permutation(len(A_d));tr=perm[:480];te=perm[480:]  # 留160测试
np.save("/home/lin/xvxinyao/ShadowForge/data/test_idx.npy",te)
aug_tr=[]
for j in tr:
    v=A_d[j].astype(np.float32);f=lg.standing_load(v)["force"]*A_m[j];aug_tr.append((v,f,np.log10(max(A_vm[j],1e-3))))
print(f"训练: hardset {len(hs)} + 零空间 {len(aug_tr)} = {len(hs)+len(aug_tr)}; 留出零空间测试 {len(te)}",flush=True)
data=hs+aug_tr
# 预缓存 xg (CPU)
print("缓存xg...",flush=True)
XG=[xg_of(d,f).cpu() for d,f,_ in data];T=torch.tensor([t for _,_,t in data],device=dev)
opt=torch.optim.AdamW(train_params,lr=3e-4,weight_decay=1e-4);lossf=torch.nn.SmoothL1Loss()
N=len(data);bs=16
for ep in range(25):
    p.train();pm=np.random.permutation(N);tot=0
    for s in range(0,N,bs):
        bi=pm[s:s+bs];xb=torch.cat([XG[k] for k in bi],0).to(dev);tb=T[bi]
        opt.zero_grad();raw=p._scale_forward(xb).squeeze(-1)
        if raw.dim()==0: raw=raw.unsqueeze(0)
        loss=lossf(raw,tb);loss.backward();opt.step();tot+=float(loss)*len(bi)
    if ep%5==0 or ep==24: print(f"  ep{ep}: loss={tot/N:.4f}",flush=True)
# 保存微调头
torch.save({"scale_branch_state_dict":{nm:getattr(p,nm).state_dict() for nm in SB},
            "attn_scale_head_state_dict":head.state_dict(),"attn_dim":c.get("attn_dim",64),"n_heads":c.get("n_heads",4)},
           "/home/lin/xvxinyao/ShadowForge/data/v40_finetuned.pt")
print("微调完成, 保存 /home/lin/xvxinyao/ShadowForge/data/v40_finetuned.pt")
print("DONE")
