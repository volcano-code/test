import sys,numpy as np,torch,glob,random,json
from pathlib import Path
ROOT="/home/lin/xvxinyao/ShadowForge";sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+"/generation")
from datasets import load_data_from_list
from voxel_ops import calibrate_tfs_projection,build_tfs_visual_hull,minimize_material_printable,make_watertight_printable,evaluate_tfs_volume
B3=ROOT+"/data/silhouettes/bench3";R=128
imgs=[b.split('/')[-1][:-4] for b in glob.glob(f"{B3}/*.png")];random.seed(2026);triples=[random.sample(imgs,3) for _ in range(15)]
res={"free":[],"fixed":[]}
for imn in triples:
    gt,_=load_data_from_list([f"{B3}/{x}.png" for x in imn],True,(R,R),torch.device("cpu"),3,False)
    gtn=[np.asarray(x).squeeze().astype(np.float32) for x in gt]
    for branch,fa in [("free",False),("fixed",True)]:
        try:
            cal=calibrate_tfs_projection(gtn,search_res=96,fix_axis=fa)
            hull,_=build_tfs_visual_hull(gtn,R,calibration=cal)
            if hull.sum()<800: continue
            v,_=minimize_material_printable(hull,bridge_radius=1);vw,_=make_watertight_printable(v,hull)
            iou=evaluate_tfs_volume(vw.astype(np.float32),gtn,cal,image_size=256)['mean_iou']
            res[branch].append(iou)
        except Exception as e: print(branch,"skip",e)
for b in ("free","fixed"):
    a=np.array(res[b]);print(f"{b:>5}: IoU={a.mean():.3f}±{a.std():.3f}  范围[{a.min():.3f},{a.max():.3f}]  n={len(a)}")
out = {
    "schema_version": "fixed_vs_free_iou_v1",
    "resolution": R,
    "n_triples_requested": len(triples),
    "branches": {
        b: {
            "n": int(len(res[b])),
            "iou_mean": float(np.mean(res[b])) if res[b] else None,
            "iou_std": float(np.std(res[b])) if res[b] else None,
            "iou_min": float(np.min(res[b])) if res[b] else None,
            "iou_max": float(np.max(res[b])) if res[b] else None,
            "ious": [float(x) for x in res[b]],
        }
        for b in ("free", "fixed")
    },
}
out_path = Path(ROOT) / "results" / "fixed_vs_free_iou.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print(f"SAVED -> {out_path}")
print("DONE")
