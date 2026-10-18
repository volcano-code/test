import sys,json,numpy as np,torch,importlib.util
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge")
from sklearn.isotonic import IsotonicRegression
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from structural.analysis.load_generator import LoadGenerator
dev=torch.device('cuda');TAU=1.3;LOGTHR=np.log10(TAU);CP="/home/lin/xvxinyao/ShadowForge/checkpoints";lg=LoadGenerator()
spec=importlib.util.spec_from_file_location("gh",ROOT+"/certification/02_geoattn_head.py");gh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gh)
p,_=load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt",device=dev);p.enable_enhanced_scale_head()
c=torch.load(f"{CP}/v40_geoattn/best.pt",map_location=dev,weights_only=False)
SB=["scale_lifting","scale_down1","scale_enc2","scale_down2","scale_bottleneck"]
for n,sd in c["scale_branch_state_dict"].items(): getattr(p,n).load_state_dict(sd)
head=gh.patch_pino_with_attn(p,attn_dim=c.get("attn_dim",64),n_heads=c.get("n_heads",4));head.load_state_dict(c["attn_scale_head_state_dict"])
for q in p.parameters(): q.requires_grad_(False)
tp=[]
for nm in SB:
    for q in getattr(p,nm).parameters(): q.requires_grad_(True);tp.append(q)
for q in head.parameters(): q.requires_grad_(True);tp.append(q)
def xg(d,f):
    x=build_x(d.astype(np.float32),f.astype(np.float32)).unsqueeze(0)
    b,cc,nx,ny,nz=x.shape;g=p._get_grid((nx,ny,nz),'cpu').expand(b,-1,-1,-1,-1);return torch.cat([x,g],1)
HARD=__import__('pathlib').Path(ROOT)/"results/shadowart_fem_hardset_v1"
dd=json.load(open("/home/lin/xvxinyao/ShadowForge/certification/pino_vs_fem_hard450.json"));paths=dd['paths'];fem=np.array(dd['fem_sf'])
ix=np.arange(len(paths));np.random.seed(0);np.random.shuffle(ix)
def aug_split(npz,seed):
    a=np.load(npz);perm=np.random.RandomState(seed).permutation(len(a['dens']));return a,perm[:440],perm[440:]
af,ftr,fte=aug_split("/home/lin/xvxinyao/ShadowForge/data/aug.npz",2)
ax,xtr,xte=aug_split("/home/lin/xvxinyao/ShadowForge/data/aug_fixed.npz",3)
# 训练样本
tr_xg=[];tr_y=[]
print("缓存训练xg...",flush=True)
for i in ix[:200]:
    d=np.load(HARD/paths[i],allow_pickle=True);tr_xg.append(xg(d['density'],d['force']));tr_y.append(np.log10(max(float(np.array(d['vm_max'])),1e-3)))
    tr_y[-1]=np.log10(max(fem[i],1e-6))  # 用sf的log? 不: target=log_vm. 修正下方
# 修正: target = log10(vm_max). hardset npz有vm_max
tr_xg=[];tr_y=[]
for i in ix[:200]:
    d=np.load(HARD/paths[i],allow_pickle=True);tr_xg.append(xg(d['density'],d['force']));tr_y.append(np.log10(max(float(np.array(d['vm_max'])),1e-3)))
for a,ids in [(af,ftr),(ax,xtr)]:
    for j in ids:
        v=a['dens'][j].astype(np.float32);f=lg.standing_load(v)["force"]*a['mult'][j];tr_xg.append(xg(v,f));tr_y.append(np.log10(max(a['vm'][j],1e-3)))
T=torch.tensor(tr_y,device=dev);N=len(tr_xg);print(f"训练{N} (hardset200+free440+fixed440)",flush=True)
opt=torch.optim.AdamW(tp,lr=3e-4,weight_decay=1e-4);lf=torch.nn.SmoothL1Loss();bs=16
for ep in range(25):
    p.train();pm=np.random.permutation(N);tot=0
    for s in range(0,N,bs):
        bi=pm[s:s+bs];xb=torch.cat([tr_xg[k] for k in bi],0).to(dev);tb=T[bi]
        opt.zero_grad();raw=p._scale_forward(xb).squeeze(-1)
        if raw.dim()==0: raw=raw.unsqueeze(0)
        l=lf(raw,tb);l.backward();opt.step();tot+=float(l)*len(bi)
    if ep%6==0 or ep==24: print(f"  ep{ep} loss={tot/N:.4f}",flush=True)
p.eval()
def raw_of(d,f):
    with torch.no_grad(): r=p._scale_forward(xg(d,f).to(dev)).squeeze(); return float(r.cpu())
# 校准(hardset+两分支train)
cr=[];cy=[]
for i in ix[:120]:
    d=np.load(HARD/paths[i],allow_pickle=True);cr.append(raw_of(d['density'],d['force']));cy.append(np.log10(max(fem[i],1e-6)))
for a,ids in [(af,ftr[:150]),(ax,xtr[:150])]:
    for j in ids:
        v=a['dens'][j].astype(np.float32);f=lg.standing_load(v)["force"]*a['mult'][j];cr.append(raw_of(v,f));cy.append(np.log10(max(a['sf'][j],1e-6)))
cr=np.array(cr);cy=np.array(cy);o=np.argsort(cr);crs,cys=cr[o],cy[o];hh=len(crs)//2
iso=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(crs[:hh],cys[:hh])
resid=np.maximum(iso.predict(crs[hh:])-cys[hh:],0);qm=np.quantile(resid,0.80)
def test_branch(a,te,tag):
    R=[];Y=[]
    for j in te:
        v=a['dens'][j].astype(np.float32);f=lg.standing_load(v)["force"]*a['mult'][j];R.append(raw_of(v,f));Y.append(np.log10(max(a['sf'][j],1e-6)))
    R=np.array(R);Y=np.array(Y);truth=Y>=LOGTHR;acc=iso.predict(R)-qm>=LOGTHR
    fp=int((acc&~truth).sum());tp_=int((acc&truth).sum());ns=int(truth.sum())
    print(f"[{tag}] n={len(te)} corr={np.corrcoef(R,Y)[0,1]:.3f} FP={fp} 认证率={tp_}/{ns}={tp_/max(ns,1):.0%}",flush=True)
print("\n=== 双分支重训后测试(α=0.2) ===")
test_branch(af,fte,"free")
test_branch(ax,xte,"fixed")
torch.save({"scale_branch_state_dict":{nm:getattr(p,nm).state_dict() for nm in SB},"attn_scale_head_state_dict":head.state_dict(),"attn_dim":c.get("attn_dim",64),"n_heads":c.get("n_heads",4)},"/home/lin/xvxinyao/ShadowForge/data/v40_finetuned_both.pt")
print("DONE")
