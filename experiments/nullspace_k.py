import sys,numpy as np,torch
sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from scipy.ndimage import label,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,evaluate_tfs_volume
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
s6=generate_binary_structure(3,1);YIELD=50e6;lg=LoadGenerator();B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3";R=64
def fem_sf99(v):
    if v.sum()==0: return 0
    L=lg.standing_load(v.astype(np.float32));o=voxel_fem_solve(v.astype(np.float32),L["force"],L["fixed_mask"],verbose=False)
    vm=o["von_mises"][v>0.5]
    return YIELD/np.nanpercentile(vm,99) if vm.size and not np.isnan(vm).all() else 0
def extrude(P,ax):  # 2D剪影沿轴ax拉伸成3D
    return np.repeat(np.expand_dims(P,ax),R,axis=ax)
def carve_axes(hull,projs,axes,maxp=30):  # 保留指定axes投影的连通保持雕除
    occ=hull.copy().astype(bool);cnt={a:occ.sum(axis=a) for a in axes}
    ch=True;p=0
    while ch and p<maxp:
        ch=False;p+=1;pad=np.pad(occ,1)
        nb=(pad[2:,1:-1,1:-1]&pad[:-2,1:-1,1:-1]&pad[1:-1,2:,1:-1]&pad[1:-1,:-2,1:-1]&pad[1:-1,1:-1,2:]&pad[1:-1,1:-1,:-2])
        for x,y,z in np.argwhere(occ&~nb):
            if not occ[x,y,z]: continue
            keep=False
            for a in axes:
                idx=(y,z) if a==0 else (x,z) if a==1 else (x,y)
                if cnt[a][idx]<=1: keep=True;break
            if keep: continue
            x0,y0,z0=max(x-1,0),max(y-1,0),max(z-1,0);win=occ[x0:x+2,y0:y+2,z0:z+2].copy()
            cx,cy,cz=x-x0,y-y0,z-z0
            if win.sum()<=1: continue
            win[cx,cy,cz]=False;lab,_=label(win,structure=s6);fl=set()
            for dx,dy,dz in[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                fx,fy,fz=cx+dx,cy+dy,cz+dz
                if 0<=fx<win.shape[0]and 0<=fy<win.shape[1]and 0<=fz<win.shape[2]and win[fx,fy,fz]:fl.add(lab[fx,fy,fz])
            if len(fl)!=1: continue
            occ[x,y,z]=False
            for a in axes:
                idx=(y,z) if a==0 else (x,z) if a==1 else (x,y); cnt[a][idx]-=1
            ch=True
    return occ
for s,imn in {"A":["birds_owl","x_lobster","bugs_ant"],"C":["reptiles_turtle","x_scorpion","birds_eagle"]}.items():
    gt,_=load_data_from_list([f"{B3}/{n}.png" for n in imn],True,(R,R),torch.device("cpu"),3,False)
    gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
    cal=calibrate_tfs_projection(gtn,search_res=96);h3,_=build_tfs_visual_hull(gtn,R,calibration=cal)
    P=[h3.max(axis=a) for a in (0,1,2)]   # 3轴投影(剪影)
    print(f"\n=== Set {s}: 欠定度 vs 零空间 (64³) ===")
    print(f"{'视图数k':>7}{'外壳材料':>9}{'最小材料':>9}{'零空间跨度':>11}{'外壳SF99':>9}{'最小SF99':>9}{'SF跨度':>8}")
    for axes in [(0,1),(0,1,2)]:
        k=len(axes)
        ext=[extrude(P[a],a) for a in axes];hull=ext[0].astype(bool)
        for e in ext[1:]: hull=hull&e.astype(bool)
        carve=carve_axes(hull,P,axes)
        Hm,Cm=int(hull.sum()),int(carve.sum())
        sfh,sfc=fem_sf99(hull),fem_sf99(carve)
        print(f"{k:>7}{Hm:>9}{Cm:>9}{Hm/max(Cm,1):>9.1f}x{sfh:>9.1f}{sfc:>9.1f}{sfh/max(sfc,0.01):>7.1f}x",flush=True)
print("DONE")
