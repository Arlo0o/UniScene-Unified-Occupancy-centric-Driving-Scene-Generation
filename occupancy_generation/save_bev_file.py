import os
import argparse
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt



# ========= Palettes from your notebook =========
MAP_PALETTE_NUPLAN = {
    "intersections": (166, 206, 227),
    "generic_drivable_areas": (166, 206, 227),
    "walkways": (227, 26, 28),
    "carpark_areas": (166, 206, 227),
    "crosswalks": (166, 206, 227),
    "lane_group_connectors": (166, 206, 227),
    "lane_groups_polygons": (166, 206, 227),
    "road_segments": (166, 206, 227),
}

OBJECT_PALETTE_NUPLAN = {
    "vehicle": (255, 158, 0),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
    "barrier": (112, 128, 144),
    "czone_sign": (255, 99, 71),
    "generic_object": (233, 150, 70),
}

LANE_PALETTE_NUPLAN = {
    "lane": (255, 30, 30),
}


def visualize_bev(canvas: np.ndarray) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()


def visualize_object_and_map(
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> np.ndarray:
    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if k >= masks.shape[0]:
            break
        if name in OBJECT_PALETTE_NUPLAN:
            canvas[masks[k] == 1, :] = OBJECT_PALETTE_NUPLAN[name]
        elif name in MAP_PALETTE_NUPLAN:
            canvas[masks[k] == 1, :] = MAP_PALETTE_NUPLAN[name]
        elif name in LANE_PALETTE_NUPLAN:
            canvas[masks[k] == 1, :] = LANE_PALETTE_NUPLAN[name]
    return canvas


# ========= Helpers =========
def load_pickle(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def try_load_bev(bev_path: str) -> Optional[np.ndarray]:
    ext = os.path.splitext(bev_path)[1].lower()
    if not os.path.exists(bev_path):
        return None
    if ext == ".npy":
        return np.load(bev_path)
    if ext == ".npz":
        npz = np.load(bev_path)
        # STRICT: only accept canonical key used in your notebook
        key = "gt_bev_masks"
        if key in npz:
            return npz[key]
        return None
    # Do not attempt to parse BEV from other formats here
    return None


def resolve_bev_path_from_token(token: str, bev_root: str) -> Optional[str]:
    candidates = [
        os.path.join(bev_root, f"{token}.npz"),
        os.path.join(bev_root, f"{token}.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def binarize_masks(masks: np.ndarray) -> np.ndarray:
    if masks.dtype != np.int64 and masks.dtype != np.int32 and masks.dtype != np.uint8:
        masks = (masks > 0).astype(np.int64)
    return masks


def render_bev_to_canvas(masks: np.ndarray, merge_like_vis: bool = False) -> np.ndarray:
    masks = binarize_masks(masks)
    # Match vis_nuplan: merge [0,1,3,4,5,6,7] into channel 1 when requested
    if merge_like_vis:
        layer_to_merge = [0, 1, 3, 4, 5, 6, 7]
        valid_idx = [i for i in layer_to_merge if i < masks.shape[0]]
        if 1 < masks.shape[0] and len(valid_idx) > 0:
            merged = np.any(masks[valid_idx, :, :], axis=0).astype(int)
            masks = masks.copy()
            masks[1, :, :] = merged
    classes_order = (
        list(MAP_PALETTE_NUPLAN.keys())
        + list(OBJECT_PALETTE_NUPLAN.keys())
        + list(LANE_PALETTE_NUPLAN.keys())
    )
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.shape[0] < len(classes_order):
        classes_order = classes_order[: masks.shape[0]]
    elif masks.shape[0] > len(classes_order):
        masks = masks[: len(classes_order), ...]
    canvas = visualize_object_and_map(masks, classes=classes_order)
    return canvas


def save_canvas(canvas: np.ndarray, out_path: str, scale: int = 1) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if scale is not None and isinstance(scale, int) and scale > 1:
        canvas = np.repeat(np.repeat(canvas, scale, axis=0), scale, axis=1)
    # Save raw pixels without figure/DPI to avoid resampling artifacts
    plt.imsave(out_path, canvas)


def process(
    pkl_path: str,
    bev_root: str,
    output_dir: Optional[str],
    limit: Optional[int] = None,
    scale: int = 1,
    merge_like_vis: bool = False,
) -> None:
    meta = load_pickle(pkl_path)
    if "infos" not in meta or not isinstance(meta["infos"], list):
        raise ValueError(f"PKL does not contain 'infos' list: {pkl_path}")

    infos: List[Dict[str, Any]] = meta["infos"]
    total = len(infos) if limit is None else min(limit, len(infos))
    processed = 0
    missing = 0

    # Default output directory: match vis behavior → "<bev_root>_vis" if not provided
    out_dir = output_dir if output_dir and len(output_dir) > 0 else f"{bev_root}_vis"

    for info in infos:
        if limit is not None and processed >= total:
            break
        token = info.get("token") if isinstance(info, dict) else None
        if not token:
            continue

        bev_path = resolve_bev_path_from_token(token, bev_root)
        if bev_path is None:
            missing += 1
            processed += 1
            continue

        bev = try_load_bev(bev_path)
        if bev is None:
            missing += 1
            processed += 1
            continue

        canvas = render_bev_to_canvas(bev, merge_like_vis=merge_like_vis)
        # Match vis filename style: <token>_bev2d.png
        out_png = os.path.join(out_dir, f"{token}.png")
        save_canvas(canvas, out_png, scale=scale)
        processed += 1

    print(
        f"Done. Processed: {processed}, Missing/failed: {missing}, Output: {out_dir}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render BEV masks to PNGs based on tokens from PKL infos."
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default="/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_val.pkl",
        help="Path to PKL containing 'infos' list with 'token' keys.",
    )
    parser.add_argument(
        "--bev_root",
        type=str,
        default="/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200",
        help="Root directory where BEV files are stored, named as <token>.(npy|npz|pkl).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output directory to save PNGs. Default: '<bev_root>_vis'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to process.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Optional integer scale to enlarge output without interpolation (e.g., 4).",
    )
    parser.add_argument(
        "--merge_like_vis",
        action="store_true",
        help="Merge layers [0,1,3,4,5,6,7] into channel 1 before rendering (match vis_nuplan).",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    process(
        pkl_path=args.pkl,
        bev_root=args.bev_root,
        output_dir=args.out,
        limit=args.limit,
        scale=args.scale,
        merge_like_vis=args.merge_like_vis,
    )


if __name__ == "__main__":
    main()


# python save_bev_file.py --pkl /data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_val.pkl --bev_root /data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200 --out /data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200_png --limit 10