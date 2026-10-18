#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path('/home/lin/xvxinyao/ShadowForge')
SRC=Path('/home/lin/xvxinyao/download_anime_character_silhouettes_balanced_friendly_20260707/images')
OUT=ROOT/'results/balanced_anime_strict_v29_48core_256delivery_20260707'
EXAMPLES=[
 ('magic_sword_mascot', ['0022_0022_0022_original_anime_silhouettes_100_2_012_magic_girl_balanced.png','0023_0023_0023_original_anime_silhouettes_100_2_013_cool_swordsman_balanced.png','0018_0018_0018_original_anime_silhouettes_100_2_008_mascot_creature_balanced.png']),
 ('sword_magic_archer', ['0033_0033_0033_original_anime_silhouettes_100_2_023_cool_swordsman_balanced.png','0032_0032_0032_original_anime_silhouettes_100_2_022_magic_girl_balanced.png','0039_0039_0039_original_anime_silhouettes_100_2_029_ranger_archer_balanced.png']),
 ('mascot_magic_swordsman', ['0048_0048_0048_original_anime_silhouettes_100_2_038_mascot_creature_balanced.png','0052_0052_0052_original_anime_silhouettes_100_2_042_magic_girl_balanced.png','0053_0053_0053_original_anime_silhouettes_100_2_043_cool_swordsman_balanced.png']),
 ('compact_magic_mascot_sword', ['0068_0068_0068_original_anime_silhouettes_100_2_058_mascot_creature_balanced.png','0062_0062_0062_original_anime_silhouettes_100_2_052_magic_girl_balanced.png','0073_0073_0073_original_anime_silhouettes_100_2_063_cool_swordsman_balanced.png']),
]
def path_for(n):
 p=SRC/n
 if not p.is_file(): raise FileNotFoundError(p)
 return str(p)
def main():
 OUT.mkdir(parents=True, exist_ok=True)
 rows=[]
 for slug,names in EXAMPLES:
  case=OUT/slug
  cmd=[sys.executable,str(ROOT/'shadowforge_pipeline/run.py'),'--silhouettes',*[path_for(n) for n in names],'--output',str(case),'--branches','free','fixed','--soft','off','--screen','v29_ensemble_acs_fem','--cert-resolution','48','--delivery-resolution','256','--min-iou','0.95','--min-saving','0.40','--max-saving','0.85','--device','cpu']
  print('RUN',slug,flush=True)
  proc=subprocess.run(cmd,cwd=str(ROOT),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  case.mkdir(parents=True,exist_ok=True); (case/'RUN_OUTPUT.txt').write_text(proc.stdout,encoding='utf-8')
  rec={'example':slug,'silhouettes':names,'returncode':proc.returncode,'manifest':str(case/'MANIFEST.json')}
  mp=case/'MANIFEST.json'
  if mp.exists():
   m=json.load(open(mp,encoding='utf-8')); d=m.get('delivery') or {}; s=d.get('structural_screen') or {}; mesh=d.get('mesh') or {}
   rec.update({'status':m.get('status'),'defer_reason':m.get('defer_reason'),'selected_core':m.get('selected_core',{}).get('candidate_id') if m.get('selected_core') else None,'candidate_counts':m.get('candidate_counts'),'mean_iou':d.get('mean_iou'),'min_iou':d.get('min_iou'),'saving':d.get('material_saved_vs_hull'),'mesh_parts':d.get('mesh_parts'),'watertight':d.get('watertight'),'manufacturing_pass':d.get('manufacturing_pass'),'struct_decided_by':s.get('decided_by'),'struct_safe':s.get('safe'),'needs_fem':s.get('needs_fem'),'sf_pino':s.get('sf_pino'),'acs_lower_sf':s.get('acs_lower_sf'),'sf_fem':s.get('sf_fem'),'mesh':mesh.get('artifact'),'delivery_artifact':d.get('artifact')})
  rows.append(rec); (OUT/'SUMMARY.partial.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
  print('DONE',slug,rec.get('status'),rec.get('mean_iou'),rec.get('saving'),rec.get('mesh_parts'),rec.get('struct_decided_by'),flush=True)
 (OUT/'SUMMARY.json').write_text(json.dumps({'schema':'balanced_anime_strict_v29_48_256_v1','rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
