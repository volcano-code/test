"""整理多样性STL: 每个shadowart组一个文件夹, 含 4 STL(fixed/free×hull/minimal) + metrics.json + 输入剪影PNG."""
import json,shutil,glob
from pathlib import Path
ROOT=Path("/home/lin/xvxinyao/ShadowForge")
SRC=ROOT/"print_ready_stls_diversity";B3=ROOT/"data/silhouettes/bench3"
OUT=SRC/"by_group";OUT.mkdir(exist_ok=True)
# (文件夹名, STL前缀, 剪影列表)
GROUPS=[
 ("n3_g1_dragon_peacock_trex","g1_x_dragon_x_peacock_x_trex",["x_dragon","x_peacock","x_trex"]),
 ("n3_g2_dragon_lobster_scorpion","g2_x_dragon_x_lobster_x_scorpion",["x_dragon","x_lobster","x_scorpion"]),
 ("n3_g3_frog_cricket_lobster","g3_reptiles_frog_bugs_cricket_x_lobster",["reptiles_frog","bugs_cricket","x_lobster"]),
 ("n3_g4_ladybug_spider_cricket","g4_x_ladybug_x_spider_bugs_cricket",["x_ladybug","x_spider","bugs_cricket"]),
 ("n3_g5_ant_dragon_shrimp","g5_bugs_ant_x_dragon_sea_shrimp",["bugs_ant","x_dragon","sea_shrimp"]),
 ("n2_g1_snail_beetle","n2_g1_sea_snail_bugs_beetle",["sea_snail","bugs_beetle"]),
 ("n2_g2_bat_owl","n2_g2_x_bat_birds_owl",["x_bat","birds_owl"]),
 ("n2_g3_parrot_peacock","n2_g3_birds_parrot_x_peacock",["birds_parrot","x_peacock"]),
]
# 读已有指标 (free n3, free n2, fixed n3); n2-fixed 用日志确认值
def load(p):
    f=SRC/p; return json.load(open(f)) if f.exists() else []
mfree_n3=load("diversity_metrics.json");mfree_n2=load("diversity_metrics_n2.json");mfix_n3=load("diversity_metrics_fixed_n3.json")
n2fix={"n2_g1_sea_snail_bugs_beetle":dict(sil_iou_hull=0.992,sil_iou_minimal=0.992,material_ratio=5.2,vol_iou_hull_vs_min=0.191),
       "n2_g2_x_bat_birds_owl":dict(sil_iou_hull=0.996,sil_iou_minimal=0.996,material_ratio=6.1,vol_iou_hull_vs_min=0.163),
       "n2_g3_birds_parrot_x_peacock":dict(sil_iou_hull=0.997,sil_iou_minimal=0.997,material_ratio=6.8,vol_iou_hull_vs_min=0.148)}
def find(lst,tag,key="triple"):
    for r in lst:
        if r.get(key)==tag or r.get("pair")==tag: return r
    return None
summary=[]
for folder,prefix,sils in GROUPS:
    d=OUT/folder;d.mkdir(exist_ok=True)
    # STL
    stls=sorted(glob.glob(str(SRC/f"{prefix}_*.stl")))
    for s in stls: shutil.copy(s,d/Path(s).name)
    # 剪影 PNG
    sdir=d/"silhouettes";sdir.mkdir(exist_ok=True)
    for nm in sils:
        src=B3/f"{nm}.png"
        if src.exists(): shutil.copy(src,sdir/f"{nm}.png")
    # 指标合并
    tag=prefix[3:] if prefix.startswith("n2_") else prefix[3:]   # 去 g#_ / n2_g#_
    rawtag="_".join(prefix.split("_")[2:]) if prefix.startswith("n2_") else "_".join(prefix.split("_")[1:])
    is_n2=prefix.startswith("n2_")
    met={"group":folder,"n":2 if is_n2 else 3,"silhouettes":sils,"n_stl":len(stls),
         "free":find(mfree_n2 if is_n2 else mfree_n3,rawtag,"pair" if is_n2 else "triple"),
         "fixed":n2fix.get(rawtag) if is_n2 else find(mfix_n3,rawtag,"triple")}
    json.dump(met,open(d/"metrics.json","w"),indent=2,ensure_ascii=False)
    summary.append({"folder":folder,"n_stl":len(stls),"n_sils":len(sils),
                    "free_iou":(met["free"] or {}).get("sil_iou_hull"),"fixed_iou":(met["fixed"] or {}).get("sil_iou_hull")})
    print(f"{folder}: {len(stls)} STL + {len(sils)} 剪影 + metrics.json",flush=True)
json.dump(summary,open(OUT/"INDEX.json","w"),indent=2)
print(f"\n整理完成 -> {OUT} ({len(GROUPS)} 组)");print("DONE")
