"""Run n=2/3 x tsf=0/1 x mr=0/1 at 250 iterations on mesh_combined2."""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PY = "/home/lin/anaconda3/envs/shadowart-revisited/bin/python"
OUT = ROOT / "output" / "n2n3_tsf_mr_iter250_20260618"
ITERS = 250
RES = 128
IMSZ = 128
FILES3 = ["Spider-Man.png", "superman.png", "batman.png"]


def parse_log(log: str) -> dict:
    export = re.findall(r"Final Export Volume: IOU=([0-9.]+), DICE=([0-9.]+)", log)
    final = re.findall(r"Final Model: IOU=([0-9.]+), DICE=([0-9.]+)", log)
    out = {"export_iou": None, "export_dice": None, "model_iou": None, "model_dice": None}
    if final:
        out["model_iou"] = float(final[-1][0])
        out["model_dice"] = float(final[-1][1])
    if export:
        out["export_iou"] = float(export[-1][0])
        out["export_dice"] = float(export[-1][1])
    return out


def read_metrics(out_dir: Path) -> dict:
    path = out_dir / "metrics_history.json"
    if not path.exists():
        return {}
    try:
        final = json.loads(path.read_text()).get("final_metrics", {})
        export = final.get("export_metrics", {})
        return {
            "metrics_export_iou": export.get("mean_iou"),
            "metrics_export_dice": export.get("mean_dice"),
            "metrics_export_strategy": export.get("export_strategy"),
            "metrics_component_count": export.get("component_count"),
        }
    except Exception as exc:
        return {"metrics_error": str(exc)}


def run_one(n: int, tsf: int, mr: int) -> dict:
    out_dir = OUT / f"n{n}_tsf{tsf}_mr{mr}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, "train.py", "cuda:0", str(out_dir), str(ITERS), "0.05", "0",
        "-sdlist", *FILES3[:n],
        "-mr", str(mr), "-tsf", str(tsf), "-tsf4", "0",
        "-cam", "ortho", "-imsz", str(IMSZ), "-res", str(RES), "-vh", "1",
        "--post_export_bundle", "0",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    (out_dir / "run.log").write_text(log)
    parsed = parse_log(log)
    parsed.update(read_metrics(out_dir))
    headline = parsed.get("metrics_export_iou") or parsed.get("export_iou") or parsed.get("model_iou")
    return {
        "n": n, "tsf": tsf, "mr": mr,
        "out_dir": str(out_dir),
        "returncode": proc.returncode,
        "sec": round(time.time() - t0, 2),
        "cmd": " ".join(cmd),
        **parsed,
        "headline_iou": headline,
        "tail_error": (proc.stderr.strip().splitlines() or [""])[-1][-240:],
    }


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for n in (2, 3):
        for tsf in (0, 1):
            for mr in (0, 1):
                key = f"n{n}_tsf{tsf}_mr{mr}"
                vals = [
                    float(r["headline_iou"])
                    for r in rows
                    if r["n"] == n and r["tsf"] == tsf and r["mr"] == mr and r.get("headline_iou") is not None
                ]
                out[key] = {
                    "n_ok": len(vals),
                    "mean_iou": round(sum(vals) / len(vals), 6) if vals else None,
                    "min_iou": round(min(vals), 6) if vals else None,
                    "max_iou": round(max(vals), 6) if vals else None,
                }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for n in (2, 3):
        for tsf in (0, 1):
            for mr in (0, 1):
                row = run_one(n, tsf, mr)
                rows.append(row)
                (OUT / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
                (OUT / "aggregate.json").write_text(json.dumps(aggregate(rows), indent=2, ensure_ascii=False))
                print(
                    f"[{time.time() - t0:7.1f}s] n={n} tsf={tsf} mr={mr} "
                    f"iou={row.get('headline_iou')} sec={row['sec']} rc={row['returncode']}",
                    flush=True,
                )
    print(json.dumps(aggregate(rows), indent=2, ensure_ascii=False), flush=True)
    print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
