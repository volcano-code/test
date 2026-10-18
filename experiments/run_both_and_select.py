"""One-click generation orchestrator: run BOTH free and fixed branches, build a
free∪fixed candidate pool, then apply the selection policy
(hard: single-6conn ∧ watertight ∧ ACS-safe(SF>=tau) ∧ IoU>=tau_iou;
 objective: max material saving; tie-break: model quality, with internal
 ε-prefer-fixed). Wires existing components only; does NOT modify train.py.

Usage:
  python run_both_and_select.py -sdlist a.png b.png c.png -out <dir> \
      [-niter 250] [-tau 1.3] [-tau_iou 0.95] [-res 256]
If <dir>/{free,fixed}/sample_0_final_voxels.npy already exist, training is skipped.
"""
import sys, os, argparse, subprocess, numpy as np, torch, importlib.util
ROOT = "/home/lin/xvxinyao/ShadowForge"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts"); sys.path.insert(0, ROOT + "/generation")
from scipy.ndimage import zoom, label, generate_binary_structure
from sklearn.isotonic import IsotonicRegression
from datasets import load_data_from_list
from structural.models.load_model import load_surrogate
from scripts.train_v26_shadowart_scale_sf import build_x
from structural.analysis.load_generator import LoadGenerator
from voxel_ops import (calibrate_tfs_projection, build_tfs_visual_hull, evaluate_tfs_volume,
                       make_watertight_printable, minimize_material_printable, analyze_connected_components)
from select_policy import select_candidate

s6 = generate_binary_structure(3, 1); dev = torch.device("cuda"); lg = LoadGenerator(); CP = ROOT + "/checkpoints"
PY = "/home/lin/anaconda3/envs/shadowart-revisited/bin/python"

# ---- PINO + ACS surrogate (verbatim setup from pipeline_top3.py) ----
spec = importlib.util.spec_from_file_location("gh", ROOT + "/certification/02_geoattn_head.py")
gh = importlib.util.module_from_spec(spec); spec.loader.exec_module(gh)
def _load(ckp):
    p, _ = load_surrogate(f"{CP}/pino_v24_ufno_dual/best_model_e10_sf0991.pt", device=dev); p.enable_enhanced_scale_head()
    c = torch.load(ckp, map_location=dev, weights_only=False)
    for n, sd in c["scale_branch_state_dict"].items(): getattr(p, n).load_state_dict(sd)
    h = gh.patch_pino_with_attn(p, attn_dim=c.get("attn_dim", 64), n_heads=c.get("n_heads", 4))
    h.load_state_dict(c["attn_scale_head_state_dict"]); p.eval(); return p
P = _load(ROOT + "/data/v40_finetuned.pt")
def _raw48(v48, M=1.0):
    f = lg.standing_load(v48.astype(np.float32))["force"] * M; x = build_x(v48.astype(np.float32), f).unsqueeze(0).to(dev)
    b, cc, nx, ny, nz = x.shape; g = P._get_grid((nx, ny, nz), x.device).expand(b, -1, -1, -1, -1)
    with torch.no_grad(): return float(P._scale_forward(torch.cat([x, g], 1)).squeeze().cpu())
_aug = np.load(ROOT + "/data/aug.npz"); _te = set(np.load(ROOT + "/data/test_idx.npy").tolist())
_tr = [j for j in range(len(_aug['dens'])) if j not in _te]
_xr = np.array([_raw48(_aug['dens'][j].astype(np.float32), _aug['mult'][j]) for j in _tr[:250]])
_yr = np.array([np.log10(max(_aug['sf'][j], 1e-6)) for j in _tr[:250]])
_iso = IsotonicRegression(increasing=(np.corrcoef(_xr, _yr)[0, 1] > 0), out_of_bounds='clip').fit(_xr, _yr)
_q = np.quantile(_iso.predict(_xr) - _yr, 0.90)
def _to48(v): return zoom(np.squeeze(v).astype(np.float32), 48.0 / np.squeeze(v).shape[0], order=0) > 0.5
def pino_sf(v):
    v48 = _to48(v); return None if v48.sum() < 40 else float(10 ** (_iso.predict([_raw48(v48)])[0] - _q))

def _train(imgs, outdir, niter, res, fixedcalib):
    npy = os.path.join(outdir, "sample_0_final_voxels.npy")
    if os.path.exists(npy): return npy
    cmd = [PY, "train.py", "cuda:0", outdir, str(niter), "0.02", "0", "-sdlist", *imgs,
           "-mr", "1", "-tsf", "1", "-fixedcalib", str(int(fixedcalib)), "-res", str(res), "--post_export_bundle", "0"]
    subprocess.run(cmd, cwd=ROOT + "/generation", check=True,
                   env={**os.environ, "OMP_NUM_THREADS": "6", "CUDA_VISIBLE_DEVICES": "0"})
    return npy

def _candidate(npy, branch, allm, fixedcalib):
    vol = np.squeeze(np.load(npy)) > 0.5
    cal = calibrate_tfs_projection(allm, search_res=96, fix_axis=fixedcalib)
    hull, _ = build_tfs_visual_hull(allm, vol.shape[0], calibration=cal); hull = hull > 0.5
    iou = float(evaluate_tfs_volume(vol.astype(np.float32), allm, cal, image_size=256)["mean_iou"])
    ncomp = int(analyze_connected_components(vol)["component_count"])
    wt = make_watertight_printable(vol, hull)[1].get("watertight", False)
    saving = 1.0 - float(vol.sum()) / max(int(hull.sum()), 1)
    nc = estimate = None
    try:
        from voxel_ops import estimate_voxel_normal_consistency
        nc = float(estimate_voxel_normal_consistency(vol))
    except Exception:
        nc = 0.0
    return {"branch": branch, "vol": vol, "iou": iou, "saving": saving, "sf": pino_sf(vol),
            "single6": ncomp == 1, "watertight": bool(wt), "quality": nc, "npy": npy}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-sdlist", "--shadow_files", nargs="+", required=True)
    ap.add_argument("-out", required=True)
    ap.add_argument("-niter", type=int, default=250)
    ap.add_argument("-res", type=int, default=256)
    ap.add_argument("-tau", type=float, default=1.3)
    ap.add_argument("-tau_iou", type=float, default=0.95)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g, _ = load_data_from_list(a.shadow_files, True, (128, 128), torch.device("cpu"), 3, False)
    allm = [np.asarray(x).squeeze().astype(np.float32) for x in g]
    cands = []
    for branch, fa in [("free", False), ("fixed", True)]:
        npy = _train(a.shadow_files, os.path.join(a.out, branch), a.niter, a.res, fa)
        cands.append(_candidate(npy, branch, allm, fa))
    for c in cands:
        print(f"[{c['branch']:5}] IoU={c['iou']:.3f} saving={c['saving']:.3f} SF={c['sf']} "
              f"single6={c['single6']} watertight={c['watertight']} quality={c['quality']:.3f}", flush=True)
    pick = select_candidate(cands, tau=a.tau, tau_iou=a.tau_iou)
    if pick is None:
        print("NO FEASIBLE CANDIDATE (none passed printable∧watertight∧SF>=tau∧IoU>=tau_iou)")
    else:
        print(f"SELECTED: branch={pick['branch']} IoU={pick['iou']:.3f} saving={pick['saving']:.3f} SF={pick['sf']} -> {pick['npy']}")

if __name__ == "__main__":
    main()
