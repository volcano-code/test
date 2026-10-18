"""Mitra-style shadow-art baseline (REIMPLEMENTATION, in the spirit of
[mitra2009shadow]; the original 2009 code/tool is not publicly available, and
Sadekar et al. (WACV'22) likewise could not obtain its output).

Method: build the silhouette-exact visual hull, then greedily carve every voxel
whose removal still keeps all three axis projections covered (projection-
preserving, NO connectivity guard). This yields a minimal silhouette-faithful
solid that, exactly as Mitra-style carving does, shatters into many disconnected
6-connected fragments ("dust") and is not directly printable.

Usage:
  python mitra_baseline.py -sdlist a.png b.png c.png -out <dir> [-res 128] [-fixedcalib 0]
Saves <dir>/sample_0_final_voxels.npy (+ optional OBJ). Returns (volume, info).
"""
import sys, os, argparse, numpy as np
sys.path.insert(0, "/home/lin/xvxinyao/ShadowForge/generation")
from scipy.ndimage import label, generate_binary_structure
from voxel_ops import (calibrate_tfs_projection, build_tfs_visual_hull,
                       analyze_connected_components, save_voxel_mesh_as_obj)
_S6 = generate_binary_structure(3, 1)


def mitra_carve(hull):
    """Greedy projection-preserving carve WITHOUT connectivity guard (Mitra-style)."""
    occ = np.asarray(hull).astype(bool).copy()
    c0 = occ.sum(0).astype(np.int64); c1 = occ.sum(1).astype(np.int64); c2 = occ.sum(2).astype(np.int64)
    changed = True
    while changed:
        changed = False
        for x, y, z in np.argwhere(occ):
            if c0[y, z] > 1 and c1[x, z] > 1 and c2[x, y] > 1:
                occ[x, y, z] = False; c0[y, z] -= 1; c1[x, z] -= 1; c2[x, y] -= 1; changed = True
    return occ


def build_mitra(silhouettes_bool, res=128, fix_axis=False):
    cal = calibrate_tfs_projection(silhouettes_bool, search_res=64, fix_axis=fix_axis)
    hull, _ = build_tfs_visual_hull(silhouettes_bool, res, calibration=cal); hull = hull > 0.5
    occ = mitra_carve(hull)
    _, ncomp = label(occ, structure=_S6)
    info = {
        "hull_voxels": int(hull.sum()),
        "carved_voxels": int(occ.sum()),
        "material_saved_ratio": float(1.0 - occ.sum() / max(int(hull.sum()), 1)),
        "component_count_6conn": int(ncomp),
        **{f"cc_{k}": v for k, v in analyze_connected_components(occ).items() if k == "component_count"},
    }
    return occ, info


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("-sdlist", "--shadow_files", nargs="+", required=True)
    ap.add_argument("-out", required=True)
    ap.add_argument("-res", type=int, default=128)
    ap.add_argument("-fixedcalib", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sils = []
    for f in a.shadow_files:
        m = cv2.resize(cv2.imread(f, 0), (256, 256), interpolation=cv2.INTER_NEAREST) > 127
        sils.append(m)
    if np.mean([s.mean() for s in sils]) > 0.5:
        sils = [~s for s in sils]  # ensure object is foreground
    vol, info = build_mitra(sils, res=a.res, fix_axis=bool(a.fixedcalib))
    np.save(os.path.join(a.out, "sample_0_final_voxels.npy"), vol[None].astype(np.float32))
    try:
        save_voxel_mesh_as_obj(vol.astype(np.uint8), os.path.join(a.out, "sample_0_output.obj"))
    except Exception as e:
        print("obj export skipped:", e)
    print("MITRA", a.out, info, flush=True)


if __name__ == "__main__":
    main()
