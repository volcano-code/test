#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, shutil, sys, time
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, generate_binary_structure

ROOT=Path('/home/lin/xvxinyao/ShadowForge')
for p in [ROOT, ROOT/'generation', ROOT/'experiments']:
    sys.path.insert(0,str(p))
from candidate_selection_batch import make_candidates, n6
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
_spec=importlib.util.spec_from_file_location('export_high_quality_mesh', str(ROOT/'generation/export_high_quality_mesh.py'))
ehqm=importlib.util.module_from_spec(_spec); assert _spec.loader is not None; _spec.loader.exec_module(ehqm)

SRC=Path('/home/lin/xvxinyao/download_anime_character_silhouettes_balanced_friendly_20260707/images')
OUT=ROOT/'results/balanced_anime_fixed_10cm_256_20260708'
R=256; PRINT_MM=100.0; NOZZLE_MM=0.4; WALL_SAFETY=2.0
IOU_MIN=0.970; MIN_VIEW=0.950; SAVE_MIN=0.40; SAVE_MAX=0.60; EROSION_MIN=0.15
S6=generate_binary_structure(3,1)
EXAMPLES=[
 ('mage_sword_mascot_A','Mage / Swordsman / Mascot',[22,23,18]),
 ('mage_sword_mascot_B','Mage / Swordsman / Mascot',[32,33,38]),
 ('mage_sword_mascot_C','Mage / Swordsman / Mascot',[52,53,48]),
 ('mage_sword_mascot_D','Mage / Swordsman / Mascot',[82,83,88]),
 ('mage_sword_mascot_E','Mage / Swordsman / Mascot',[102,103,98]),
 ('school_mage_mascot','Schoolgirl / Mage / Mascot',[6,22,28]),
 ('ninja_mage_mascot','Ninja / Mage / Mascot',[4,32,18]),
 ('hero_mage_android','Hero / Mage / Android',[11,22,20]),
 ('cat_sword_mascot','Cat-ear / Swordsman / Mascot',[16,23,18]),
 ('mecha_mage_sword','Mecha / Mage / Swordsman',[15,22,23]),
]

def file_for(idx:int)->Path:
    hits=sorted(SRC.glob(f'{idx:04d}_*.png'))
    if len(hits)!=1: raise FileNotFoundError((idx,hits))
    return hits[0]

def save_mask(mask,path):
    Image.fromarray(((np.asarray(mask)>0.5).astype(np.uint8)*255)).resize((512,512), Image.Resampling.NEAREST).save(path)

def mesh_clean_scale(path:Path)->trimesh.Trimesh:
    mesh=trimesh.load(str(path), force='mesh')
    mesh.merge_vertices(); mesh.update_faces(mesh.unique_faces())
    parts=mesh.split(only_watertight=False)
    if parts: mesh=max(parts, key=lambda x: len(x.faces))
    trimesh.repair.fill_holes(mesh)
    if mesh.volume < 0: mesh.invert()
    trimesh.repair.fix_normals(mesh)
    mesh.apply_scale(PRINT_MM/max(float(np.asarray(mesh.extents).max()),1e-9))
    mesh.export(str(path)); return mesh

def erosion_survival(volume):
    vol=np.asarray(volume).astype(bool)
    return float(binary_erosion(vol,S6).sum()/max(int(vol.sum()),1))

def contact_sheet(title, inputs6, preds6, out):
    cell=130; gap=18; w=6*cell+7*gap; h=2*cell+4*gap+28
    canvas=Image.new('RGB',(w,h),'white'); draw=ImageDraw.Draw(canvas); draw.text((gap,6),title,fill=(0,0,0))
    for i in range(6):
        x=gap+i*(cell+gap); y1=gap+28; y2=y1+cell+gap
        inp=Image.fromarray((np.asarray(inputs6[i])>0.5).astype(np.uint8)*255).resize((cell,cell), Image.Resampling.NEAREST)
        prd=Image.fromarray((np.asarray(preds6[i])>0.5).astype(np.uint8)*255).resize((cell,cell), Image.Resampling.NEAREST)
        canvas.paste(inp.convert('RGB'),(x,y1)); canvas.paste(prd.convert('RGB'),(x,y2))
        draw.rectangle((x,y1,x+cell,y1+cell),outline=(120,120,120)); draw.rectangle((x,y2,x+cell,y2+cell),outline=(120,120,120))
    canvas.save(out)

def run_one(slug,title,idxs):
    t0=time.time(); paths=[str(file_for(i)) for i in idxs]; d=OUT/slug
    (d/'inputs6').mkdir(parents=True,exist_ok=True); (d/'projections6').mkdir(parents=True,exist_ok=True)
    sils,_=load_data_from_list(paths, True, (R,R), torch.device('cpu'), 3, False)
    inputs6=[np.asarray(x).squeeze().astype(np.float32) for x in sils[:6]]; masks=inputs6[:3]
    cands,aux=make_candidates(masks,R,True,print_mm=PRINT_MM,nozzle_mm=NOZZLE_MM,wall_safety=WALL_SAFETY)
    gt6=masks+[np.fliplr(m) for m in masks]; hull_vox=max(int(aux['hull'].sum()),1)
    rows=[]
    for cname,vol in cands.items():
        vol=vol.astype(bool); ev=evaluate_tfs_volume(vol.astype(np.float32), gt6, aux['calibration'], image_size=512)
        try:
            watertight=bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight)
        except Exception:
            watertight=False
        rows.append({'candidate':cname,'mean_iou':float(ev['mean_iou']),'min_iou':float(min(ev['view_ious'][:3])),'view_ious6':[float(x) for x in ev['view_ious'][:6]],'saving':float(1-int(vol.sum())/hull_vox),'single':bool(n6(vol)==1),'watertight':watertight,'erosion_survival':erosion_survival(vol),'_vol':vol,'_ev':ev})
    gate=[r for r in rows if r['mean_iou']>=IOU_MIN and r['min_iou']>=MIN_VIEW and SAVE_MIN<=r['saving']<=SAVE_MAX and r['single'] and r['watertight'] and r['erosion_survival']>=EROSION_MIN]
    chosen=max(gate, key=lambda r:(-abs(r['saving']-0.50), r['mean_iou'], r['erosion_survival'])) if gate else max(rows, key=lambda r:(r['mean_iou'], r['single'], r['watertight'], r['erosion_survival']))
    pred3=[np.asarray(x) for x in chosen['_ev']['pred_masks'][:3]]; preds6=pred3+[np.fliplr(x) for x in pred3]
    for i,a in enumerate(inputs6): save_mask(a,d/'inputs6'/f'input_view{i}.png')
    for i,a in enumerate(preds6): save_mask(a,d/'projections6'/f'projection_view{i}.png')
    contact_sheet(title,inputs6,preds6,d/f'{slug}_six_input_vs_projection.png')
    stem=f"{slug}_{chosen['candidate']}"; npy=d/f'{stem}_voxels256.npy'; stl=d/f'{stem}_10cm_256.stl'
    np.save(npy, chosen['_vol'].astype(np.uint8)); ehqm.export_marching_cubes(chosen['_vol'].astype(bool), stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
    mesh=mesh_clean_scale(stl); mesh_parts=int(len(mesh.split(only_watertight=False)))
    final_ok=bool(chosen in gate and mesh.is_watertight and mesh_parts==1)
    return {'example':slug,'group_name':title,'indices':idxs,'input_paths':paths,'resolution':R,'print_mm':PRINT_MM,'fixed':True,'accepted':final_ok,'selected':{'candidate':chosen['candidate'],'mean_iou':round(chosen['mean_iou'],6),'min_iou':round(chosen['min_iou'],6),'view_ious6':[round(float(x),6) for x in chosen['view_ious6']],'saving':round(chosen['saving'],6),'single':chosen['single'],'watertight':chosen['watertight'],'mesh_watertight':bool(mesh.is_watertight),'mesh_parts':mesh_parts,'erosion_survival':round(chosen['erosion_survival'],6),'stl':str(stl),'voxels':str(npy),'six_view_preview':str(d/f'{slug}_six_input_vs_projection.png')},'all_candidates':[{k:(round(v,6) if isinstance(v,float) else v) for k,v in r.items() if not k.startswith('_')} for r in rows], 'sec':round(time.time()-t0,1)}

def main():
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True,exist_ok=True); recs=[]
    for slug,title,idxs in EXAMPLES:
        rec=run_one(slug,title,idxs); recs.append(rec); s=rec['selected']
        print(f"{slug}: accepted={rec['accepted']} {s['candidate']} IoU={s['mean_iou']} min={s['min_iou']} saving={s['saving']} mesh_parts={s['mesh_parts']} watertight={s['mesh_watertight']}", flush=True)
        (OUT/'MANIFEST.partial.json').write_text(json.dumps(recs,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'schema':'balanced_anime_fixed_10cm_256_v1','source_dir':str(SRC),'n_examples':len(recs),'accepted':sum(r['accepted'] for r in recs),'examples':recs}
    (OUT/'MANIFEST.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',OUT)
if __name__=='__main__': main()
