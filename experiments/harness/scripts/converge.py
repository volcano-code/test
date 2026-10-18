import sys
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import zoom
SF=Path("/home/lin/xvxinyao/ShadowForge")
for p in [SF, SF/"generation", SF/"experiments"]: sys.path.insert(0,str(p))
from datasets import load_data_from_list
from candidate_selection_batch import make_candidates, keep_largest
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
YIELD=50e6; B=str(SF/"data/silhouettes/bench3")
tri=['x_lobster','x_bat','reptiles_frog']
# 固定几何: 在 R=48 生成 watertight, 之后只【上采样同一形状】到更高网格
s,_=load_data_from_list([f'{B}/{n}.png' for n in tri],True,(48,48),torch.device('cpu'),3,False); m=[np.asarray(x).squeeze().astype(np.float32) for x in s]
c,_=make_candidates(m,48,True); base=keep_largest(c['watertight_export']).astype(bool)
print(f'固定几何=lobster watertight@48 (occ={int(base.sum())}), 上采样到不同网格, 物理尺寸固定(L=1)')
print('R | SFmax | SF99 | SF95 | SF90 | (同一物体, 仅网格变细)')
for R in (48,64,96,128):
    v = base.astype(np.float32) if R==48 else (zoom(base.astype(np.float32),R/48,order=0)>0.5).astype(np.float32)
    cell=1.0/R  # 物体边长固定=1
    L=LoadGenerator(cell_size=cell).standing_load(v)
    fo=voxel_fem_solve(v,L['force'],L['fixed_mask'],cell_size=cell,verbose=False); vm=fo['von_mises'][v.astype(bool)]; vm=vm[np.isfinite(vm)]
    f=lambda p: YIELD/max(np.percentile(vm,p),1e-9)
    print(f'{R:3d} | {YIELD/max(vm.max(),1e-9):7.2f} | {f(99):7.2f} | {f(95):7.2f} | {f(90):7.2f}',flush=True)
print("DONE")
