import sys,gc,numpy as np
sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"mesh_combined2")
from scipy.ndimage import zoom,label,generate_binary_structure
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
YIELD=50e6;TAU=1.3;lg=LoadGenerator();s6=generate_binary_structure(3,1)
print(f"口径: 64^3(同PINO/ACCS分辨率) + standing_load + 50MPa, tau={TAU}",flush=True)
print(f"{'Set':>4}{'occ64':>7}{'6conn':>6}{'SF_max':>8}{'SF_p99':>8}{'pass':>5}",flush=True)
for s in "ABCD":
    v=np.load(f"/tmp/printable_{s}.npy").astype(bool)
    v64=zoom(v.astype(np.float32),0.5,order=0)>0.5
    _,n6=label(v64,structure=s6)
    L=lg.standing_load(v64.astype(np.float32))
    o=voxel_fem_solve(v64.astype(np.float32),L["force"],L["fixed_mask"],verbose=False)
    vms=o["von_mises"][v64>0.5]
    if vms.size==0 or np.isnan(vms).all(): print(f"{s:>4} empty/degenerate",flush=True);continue
    mx=YIELD/np.nanmax(vms);p99=YIELD/np.nanpercentile(vms,99)
    print(f"{s:>4}{int(v64.sum()):>7}{n6:>6}{mx:>8.2f}{p99:>8.2f}{str(p99>=TAU):>5}",flush=True)
    del o,L,vms;gc.collect()
print("DONE",flush=True)
