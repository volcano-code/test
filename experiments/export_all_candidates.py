"""导出 heroes 全部候选(几何零空间完整材料阶梯 free+fixed). 每级 STL + manifest(材料%/IoU/FEM_SF)."""
import sys,json,numpy as np,torch
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+"/generation")
from pathlib import Path
from scipy.ndimage import label,generate_binary_structure,zoom
s6=generate_binary_structure(3,1);YIELD=50e6;MM=210.0
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,evaluate_tfs_volume,minimize_material_printable,make_watertight_printable,sdf_watertight_mesh
from safe_minimize import build_ladder
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.fem_solver import voxel_fem_solve
lg=LoadGenerator()
OUT=Path(ROOT+"/print_ready_stls_diversity/by_group/n3_heroes_spiderman_superman_batman/all_candidates");OUT.mkdir(parents=True,exist_ok=True)
HER=["Spider-Man","superman","batman"];files=[f"{ROOT}/generation/{h}.png" for h in HER]
g,_=load_data_from_list(files,True,(128,128),torch.device("cpu"),3,False);allm=[np.asarray(x).squeeze().astype(np.float32) for x in g]
def femsf(v):
    v48=zoom(np.squeeze(v).astype(np.float32),48.0/np.squeeze(v).shape[0],order=0)>0.5
    if v48.sum()<40:return None
    L=lg.standing_load(v48.astype(np.float32));o=voxel_fem_solve(v48.astype(np.float32),L["force"],L["fixed_mask"],verbose=False)
    vm=o["von_mises"][v48>0.5];return round(YIELD/np.nanmax(vm),2) if vm.size and not np.isnan(vm).all() else None
manifest=[]
for mode,fa in [("free",False),("fixed",True)]:
    cal=calibrate_tfs_projection(allm,search_res=96,fix_axis=fa);hull,_=build_tfs_visual_hull(allm,128,calibration=cal)
    mn,_=minimize_material_printable(hull,bridge_radius=1);ladder=build_ladder(hull,mn,8)
    Hocc=int(make_watertight_printable(hull,hull)[0].sum())
    print(f"\n=== {mode}: {len(ladder)} 个候选(材料阶梯) ===",flush=True)
    for i,v in enumerate(ladder):
        vw,_=make_watertight_printable(v,hull)
        iou=evaluate_tfs_volume(np.squeeze(vw).astype(np.float32),allm,cal,image_size=256)['mean_iou']
        sf=femsf(vw);matp=round(100*int(np.squeeze(vw).sum())/Hocc,1)
        m=sdf_watertight_mesh((np.squeeze(vw)>0.5).astype(np.uint8),0.6);ext=(m.vertices.max(0)-m.vertices.min(0)).max();m.vertices*=MM/ext
        fn=f"{mode}_cand{i:02d}_mat{int(matp)}_sf{sf}.stl";m.export(str(OUT/fn))
        rec=dict(mode=mode,idx=i,material_pct=matp,iou=round(float(iou),3),fem_sf=sf,single=int(label(np.squeeze(vw)>0.5,s6)[1]==1),file=fn)
        manifest.append(rec);print(f"  cand{i:02d}: 材料{matp:>5}% IoU={iou:.3f} SF={sf}",flush=True)
json.dump(manifest,open(OUT/"manifest.json","w"),indent=2)
print(f"\n全部导出 -> {OUT} (共 {len(manifest)} 个候选)");print("DONE")
