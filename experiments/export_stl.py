import sys,numpy as np,torch
sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge");sys.path.insert(0,"/home/lin/xvxinyao/ShadowForge/generation")
from scipy.ndimage import label,generate_binary_structure
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,minimize_material_printable,make_watertight_printable,sdf_watertight_mesh,evaluate_tfs_volume
s6=generate_binary_structure(3,1);R=128;B3="/home/lin/xvxinyao/ShadowForge/data/silhouettes/bench3"
TRIPLES={"owl_lobster_ant":["birds_owl","x_lobster","bugs_ant"],
         "turtle_scorpion_eagle":["reptiles_turtle","x_scorpion","birds_eagle"],
         "dragon_bat_spider":["x_dragon","x_bat","x_spider"],
         "peacock_frog_beetle":["x_peacock","reptiles_frog","bugs_beetle"]}
print(f"{'三元组':>26}{'省材%':>7}{'IoU':>7}{'6连通':>6}{'水密':>6}{'面数':>7}")
for name,imn in TRIPLES.items():
    try:
        gt,_=load_data_from_list([f"{B3}/{n}.png" for n in imn],True,(R,R),torch.device("cpu"),3,False)
        gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
        cal=calibrate_tfs_projection(gtn,search_res=96);hull,_=build_tfs_visual_hull(gtn,R,calibration=cal);H=int(hull.sum())
        v,_=minimize_material_printable(hull,bridge_radius=1)
        vw,info=make_watertight_printable(v,hull)  # 保证水密
        m=sdf_watertight_mesh(vw,smooth_sigma=0.6)
        # 缩放到 210mm 打印尺寸
        m.apply_scale(210.0/R)
        path=f"/tmp/print_stls/{name}.stl";m.export(path)
        iou=evaluate_tfs_volume(vw.astype(np.float32),gtn,cal,image_size=256)['mean_iou']
        _,n6=label(vw,structure=s6)
        print(f"{name:>26}{info['material_saved_ratio']*100:6.1f}%{iou:>7.3f}{n6:>6}{str(m.is_watertight):>6}{len(m.faces):>7}",flush=True)
    except Exception as e: print(f"{name}: ERR {str(e)[:40]}",flush=True)
import glob;print("\n导出STL:",glob.glob("/tmp/print_stls/*.stl"))
print("DONE")
