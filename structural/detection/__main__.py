"""CLI: python -m structural.detection <voxel_file> [options]

Examples:
    python -m structural.detection test_datasets/modelnet40/model_0002/voxels.npy
    python -m structural.detection part.obj --sf-target 20 --repair
    python -m structural.detection part.npy --load-case grip --no-fem
"""
import argparse
import json
import sys
from pathlib import Path

from structural.detection import StrengthDetector, DetectorConfig


def main():
    ap = argparse.ArgumentParser(prog="structural.detection")
    ap.add_argument("input", help="Path to voxel .npy/.npz or surface mesh .obj/.stl")
    ap.add_argument("--sf-target", type=float, default=15.0, help="Required safety factor")
    ap.add_argument("--load-case", default="standing", choices=["standing", "grip"])
    ap.add_argument("--no-fem", action="store_true", help="Disable FEM fallback in UNCERTAIN band")
    ap.add_argument("--repair", action="store_true", help="If FAIL, run Stage C repair and re-detect")
    ap.add_argument("--output", default=None, help="Optional JSON report path")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"ERROR: input not found: {inp}")

    det = StrengthDetector(DetectorConfig())

    if args.repair:
        out = det.detect_and_repair(
            inp, sf_target=args.sf_target, load_case=args.load_case,
        )
    else:
        r = det.detect_file(
            inp, sf_target=args.sf_target, load_case=args.load_case,
            fem_fallback=not args.no_fem,
        )
        out = {"detection": r.to_dict()}

    print("\n" + "=" * 60)
    print(f"DETECTION REPORT — {inp}")
    print("=" * 60)
    d = out.get("final_detection") or out.get("detection")
    print(f"  status         : {d['status']}")
    print(f"  sf_target      : {d['sf_target']}")
    print(f"  sf_pred_raw    : {d['sf_pred_raw']:.2f}  (raw PINO calibrator)")
    print(f"  sf_pred        : {d['sf_pred']:.2f}  (bias-corrected, used for decision)")
    if d.get("sf_confirmed") is not None:
        print(f"  sf_confirmed   : {d['sf_confirmed']:.2f}  (voxel FEM)")
    print(f"  pass_threshold : {d['pass_threshold']:.2f}")
    print(f"  fail_threshold : {d['fail_threshold']:.2f}")
    print(f"  decision_path  : {d['decision_path']}")
    print(f"  ood_score      : {d['ood_score']:.3f}")
    print(f"  load_case      : {d['load_case']}")
    print(f"  time_ms        : {d['time_ms']:.0f}")
    print(f"  next_action    : {d['next_action']}")

    if out.get("repair"):
        r = out["repair"]
        print(f"\n  [repair] sf_final={r.get('sf_final'):.2f}  "
              f"iou={r.get('iou_final'):.3f}  "
              f"vol_ratio={r.get('vol_ratio_final'):.3f}  "
              f"n_iters={int(r.get('n_iters'))}  "
              f"reached={r.get('sf_target_reached')}")
        obj_ex = out.get("obj_export")
        if obj_ex:
            chosen = obj_ex.get("chosen")
            if chosen:
                print(f"  [obj-export] SAFE: sf_revox={chosen['sf_revox']:.2f}  "
                      f"vol_ratio={chosen['vol_ratio']:.3f}  "
                      f"params={chosen['tag']}  obj={chosen.get('final_obj')}")
            else:
                print(f"  [obj-export] FAILED: 0/{obj_ex.get('n_searched',0)} candidates safe")
        fd = out["final_detection"]
        print(f"  [final] status={fd['status']}  "
              f"sf_confirmed={fd.get('sf_confirmed')}  "
              f"decision={fd.get('decision_path')}  "
              f"next_action={fd.get('next_action')}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  report written: {args.output}")


if __name__ == "__main__":
    main()
