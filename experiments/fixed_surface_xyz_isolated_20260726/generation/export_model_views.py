"""
export_model_views.py
Export rendered views and silhouette views for an arbitrary OBJ/NPY model.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from post_export import (
    _create_projection_family_panels,
    _create_silhouette_keyviews_gif,
    _create_sixview_panel,
    _get_view_angles,
    _get_view_labels,
    _mesh_from_binary_volume,
    _mesh_from_obj,
    _render_mesh_rgb_and_masks,
    _render_turntable,
    _resolve_device,
)


def _load_mesh(model_path: str, device):
    suffix = Path(model_path).suffix.lower()
    if suffix == ".obj":
        model_p = Path(model_path)
        sibling_voxels = model_p.with_name(model_p.stem.replace("_output", "_final_voxels") + ".npy")
        if sibling_voxels.exists():
            volume = np.load(sibling_voxels)
            mesh = _mesh_from_binary_volume(volume, device)
        else:
            mesh = _mesh_from_obj(model_path, device)
    elif suffix == ".npy":
        volume = np.load(model_path)
        mesh = _mesh_from_binary_volume(volume, device)
    else:
        raise ValueError(f"Unsupported model format: {suffix}. Use .obj or .npy")
    if mesh is None:
        raise RuntimeError(f"Failed to load model from {model_path}")
    return mesh


def export_model_views(
    model_path: str,
    output_dir: str,
    device: str = "cpu",
    projection_mode: str = "tsf",
    camera_mode: str = "ortho",
    mirror_mode: bool = True,
    num_views: int = 3,
    image_size: int = 256,
    rotation_gif: bool = False,
    rotation_frames: int = 72,
    rotation_fps: int = 18,
    keyview_pause_seconds: float = 10.0,
):
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    device_t = _resolve_device(device)
    model_path_obj = Path(model_path).resolve()
    mesh = _load_mesh(model_path, device_t)
    geometry_type = "voxel" if model_path_obj.suffix.lower() == ".npy" else "mesh"
    voxel_path = str(model_path_obj) if geometry_type == "voxel" else None
    obj_path = str(model_path_obj) if model_path_obj.suffix.lower() == ".obj" else None
    if obj_path is None and voxel_path is not None:
        sibling_obj = model_path_obj.with_name(model_path_obj.stem.replace("_final_voxels", "_output") + ".obj")
        if sibling_obj.exists():
            obj_path = str(sibling_obj)

    summary = {
        "run_id": model_path_obj.stem,
        "source_dir": str(output_path),
        "sample_id": 1,
        "geometry_type": geometry_type,
        "projection_mode": projection_mode,
        "camera_mode": camera_mode,
        "mirror_mode": bool(mirror_mode),
        "view_count": 6 if projection_mode == "tsf" and mirror_mode else (3 if projection_mode == "tsf" else (num_views * 2 if mirror_mode else num_views)),
        "artifact_paths": {"final_views": [], "voxels": voxel_path, "obj": obj_path},
        "aggregate_metrics": {"mean_iou": 0.0},
    }
    if projection_mode == "tsf4v":
        summary["view_count"] = 8 if mirror_mode else 4

    angles = _get_view_angles(summary)
    labels = _get_view_labels(summary)
    rgbs, masks = _render_mesh_rgb_and_masks(
        mesh,
        angles,
        device=device_t,
        image_size=image_size,
        camera_mode=camera_mode,
    )

    base_name = Path(model_path).stem
    for idx, (rgb, mask) in enumerate(zip(rgbs, masks)):
        render_path = output_path / f"{base_name}_render_view_{idx}.png"
        sil_path = output_path / f"{base_name}_silhouette_view_{idx}.png"
        cv2.imwrite(str(render_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sil_path), mask)
        summary["artifact_paths"]["final_views"].append(str(sil_path))

    hero = rgbs[0] if rgbs else None
    artistic_view_limit = min(len(masks), 8 if projection_mode == "tsf4v" else 6)
    silhouettes = [mask for mask in masks[:artistic_view_limit]]
    silhouette_labels = labels[:artistic_view_limit]
    base_view_count = num_views if projection_mode != "tsf4v" else 4
    if mirror_mode and artistic_view_limit >= base_view_count * 2:
        base_view_count = artistic_view_limit // 2
    else:
        base_view_count = min(base_view_count, artistic_view_limit)
    artistic_primary_path = output_path / f"{base_name}_projection_artistic_primary.png"
    artistic_opposite_path = output_path / f"{base_name}_projection_artistic_opposite.png"
    artistic_path = output_path / f"{base_name}_projection_artistic.png"
    artistic_paths = _create_projection_family_panels(
        hero_rgb=hero,
        silhouettes=silhouettes,
        silhouette_labels=silhouette_labels,
        projection_mode=projection_mode,
        primary_title=f"{base_name} Projection Sculpture",
        primary_subtitle="views: " + ", ".join(silhouette_labels[:base_view_count]),
        primary_out_path=artistic_primary_path,
        secondary_out_path=artistic_opposite_path,
        combined_out_path=artistic_path,
        base_view_count=base_view_count,
        view_angles=angles[:artistic_view_limit],
        primary_hero_rgb=rgbs[0] if rgbs else hero,
        secondary_hero_rgb=rgbs[base_view_count] if len(rgbs) > base_view_count else hero,
    )

    sixview_path = output_path / f"{base_name}_projection_sixview.png"
    _create_sixview_panel(summary, rgbs[:6], sixview_path)
    keyviews_path = output_path / f"{base_name}_silhouette_keyviews.gif"
    keyviews_gif = _create_silhouette_keyviews_gif(
        summary,
        rgbs[:6],
        keyviews_path,
        device=device_t,
        fps=max(6, int(rotation_fps)),
        transition_frames=8,
        pause_seconds=float(keyview_pause_seconds),
    )

    rotation_path = None
    if rotation_gif:
        rotation_path = output_path / f"{base_name}_rotation.gif"
        _render_turntable(
            mesh,
            device=device_t,
            frames=rotation_frames,
            fps=rotation_fps,
            camera_mode=camera_mode,
            out_path=rotation_path,
        )

    manifest_lines = [
        f"model: {Path(model_path).resolve()}",
        f"projection_mode: {projection_mode}",
        f"camera_mode: {camera_mode}",
        f"mirror_mode: {bool(mirror_mode)}",
        f"view_count: {len(angles)}",
        f"keyview_pause_seconds: {float(keyview_pause_seconds):.2f}",
    ]
    for idx, label in enumerate(labels):
        manifest_lines.append(f"view_{idx}: {label}")
    manifest_lines.append(f"projection_artistic: {artistic_path}")
    if artistic_paths.get("primary"):
        manifest_lines.append(f"projection_artistic_primary: {artistic_paths['primary']}")
    if artistic_paths.get("secondary"):
        manifest_lines.append(f"projection_artistic_opposite: {artistic_paths['secondary']}")
    manifest_lines.append(f"projection_sixview: {sixview_path}")
    manifest_lines.append(f"silhouette_keyviews_gif: {keyviews_gif}")
    if rotation_path is not None:
        manifest_lines.append(f"rotation_gif: {rotation_path}")

    manifest_path = output_path / f"{base_name}_view_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "artistic": str(artistic_path),
        "artistic_primary": artistic_paths.get("primary"),
        "artistic_opposite": artistic_paths.get("secondary"),
        "sixview": str(sixview_path),
        "keyviews_gif": keyviews_gif,
        "rotation_gif": str(rotation_path) if rotation_path is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Export rendered model views and silhouette views for a given OBJ/NPY model")
    parser.add_argument("--model", required=True, help="Path to .obj or .npy model")
    parser.add_argument("--output_dir", required=True, help="Directory for exported images")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--projection_mode", default="tsf", choices=["tsf", "ring", "tsf4v"])
    parser.add_argument("--camera_mode", default="ortho", choices=["ortho", "perp"])
    parser.add_argument("--mirror_mode", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=3, help="Base view count for ring mode")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--rotation_gif", type=int, default=0)
    parser.add_argument("--rotation_frames", type=int, default=72)
    parser.add_argument("--rotation_fps", type=int, default=18)
    parser.add_argument("--keyview_pause_seconds", type=float, default=10.0)
    args = parser.parse_args()

    export_model_views(
        model_path=args.model,
        output_dir=args.output_dir,
        device=args.device,
        projection_mode=args.projection_mode,
        camera_mode=args.camera_mode,
        mirror_mode=bool(args.mirror_mode),
        num_views=args.num_views,
        image_size=args.image_size,
        rotation_gif=bool(args.rotation_gif),
        rotation_frames=args.rotation_frames,
        rotation_fps=args.rotation_fps,
        keyview_pause_seconds=args.keyview_pause_seconds,
    )


if __name__ == "__main__":
    main()
