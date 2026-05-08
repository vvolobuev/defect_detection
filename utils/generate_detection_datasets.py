import argparse
import importlib.util
import shutil
from pathlib import Path

import cv2
import numpy as np


class FrameItem:
    def __init__(self, scene_dir, scene_id, image_id):
        self.scene_dir = scene_dir
        self.scene_id = scene_id
        self.image_id = image_id


def load_dent_module(module_path):
    spec = importlib.util.spec_from_file_location("apply_realistic_dents", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def depth_to_heat(depth_u16):
    valid = depth_u16 > 0
    norm = np.zeros_like(depth_u16, dtype=np.uint8)
    if np.any(valid):
        d = depth_u16.astype(np.float32)
        dmin = d[valid].min()
        dmax = d[valid].max()
        inv = (dmax - d[valid]) / (dmax - dmin + 1e-6)                
        norm[valid] = np.clip(inv * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    heat[~valid] = 0
    return heat


def contours_boxes(mask):
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 25:
            continue
        out.append((x, y, w, h, area))
    return out


def boxes_to_yolo(boxes, width, height):
    lines = []
    for x, y, w, h, _ in boxes:
        xc = (x + w / 2.0) / width
        yc = (y + h / 2.0) / height
        wn = w / width
        hn = h / height
        lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return lines


def build_soft_rgb_and_depth_heat(rgb, depth, mask, rgb_anom):
    m = (mask > 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
    d = dist_in / dist_in.max() if dist_in.max() > 0 else dist_in

                                    
    profile_rgb = np.clip(0.42 * m.astype(np.float32) + 0.58 * np.power(np.clip(d, 0.0, 1.0), 0.95), 0.0, 1.0)
    profile_rgb = cv2.GaussianBlur(profile_rgb, (0, 0), 1.2)
    rgb_f = rgb_anom.astype(np.float32) / 255.0
    dark_factor = (1.0 - 0.28 * profile_rgb)[:, :, None]
    rgb_soft = np.clip(rgb_f * dark_factor, 0.0, 1.0)
    rgb_soft = (rgb_soft * 255).astype(np.uint8)

                                      
    depth_f = depth.astype(np.float32)
    valid = depth_f > 0
    span = max(1.0, float(depth_f[valid].max() - depth_f[valid].min())) if np.any(valid) else 100.0
    profile_d = np.clip(0.46 * m.astype(np.float32) + 0.54 * np.power(np.clip(d, 0.0, 1.0), 1.05), 0.0, 1.0)
    profile_d = cv2.GaussianBlur(profile_d, (0, 0), 1.5)
    depth_anom = depth_f.copy()
    depth_anom[valid] = depth_f[valid] + (0.075 * span) * profile_d[valid]
    depth_anom = np.clip(depth_anom, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    return rgb_soft, depth_to_heat(depth_anom)


def discover_frames(scenes_root):
    items = []
    for scene_dir in sorted([p for p in scenes_root.iterdir() if p.is_dir()]):
        scene_id = scene_dir.name
        rgb_dir = scene_dir / "rgb"
        depth_dir = scene_dir / "depth"
        mask_visib_dir = scene_dir / "mask_visib"
        if not (rgb_dir.exists() and depth_dir.exists() and mask_visib_dir.exists()):
            continue
        for img_path in sorted(rgb_dir.glob("*.png")):
            image_id = img_path.stem
            if (depth_dir / f"{image_id}.png").exists():
                items.append(FrameItem(scene_dir=scene_dir, scene_id=scene_id, image_id=image_id))
    return items


def split_items(items, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    shuffled = [items[i] for i in idx]
    n = len(shuffled)
    n_train = int(n * 0.7)
    n_valid = int(n * 0.2)
    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test": shuffled[n_train + n_valid:],
    }


def write_data_yaml(ds_root):
    (ds_root / "data.yaml").write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                "nc: 1",
                "names: ['defect']",
            ]
        ),
        encoding="utf-8",
    )


def prepare_dataset_root(ds_root, overwrite):
    if ds_root.exists() and overwrite:
        shutil.rmtree(ds_root)
    ds_root.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        (ds_root / split / "images").mkdir(parents=True, exist_ok=True)
        (ds_root / split / "labels").mkdir(parents=True, exist_ok=True)


def build_datasets(scenes_root, output_root, dent_module_path, seed, overwrite, max_retries):
    items = discover_frames(scenes_root)
    if not items:
        raise RuntimeError(f"No valid frames found under {scenes_root}")

    splits = split_items(items, seed=seed)

    rgb_ds = output_root / "defects_rgb_det"
    depth_ds = output_root / "defects_depth_heat_det"
    prepare_dataset_root(rgb_ds, overwrite=overwrite)
    prepare_dataset_root(depth_ds, overwrite=overwrite)

    dent_mod = load_dent_module(dent_module_path)
    rng = np.random.default_rng(seed + 1000)

    for split_name, split_items in splits.items():
        for item in split_items:
            rgb = cv2.imread(str(item.scene_dir / "rgb" / f"{item.image_id}.png"), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(item.scene_dir / "depth" / f"{item.image_id}.png"), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                continue

            accepted = False
            best_rgb_anom = None
            best_mask = None
            best_boxes = None

            for _ in range(max_retries):
                target_anoms = int(rng.integers(1, 4))        
                try:
                    rgb_anom, mask = dent_mod.process_image(
                        scene_dir=item.scene_dir,
                        image_id=item.image_id,
                        rng=rng,
                        objects_per_image=target_anoms,
                        dents_per_object=1,
                        mode="box_crumple",
                    )
                except Exception:
                    continue
                boxes = contours_boxes(mask)
                k = len(boxes)
                if k < 1 or k > 3:
                    continue
                if k == 3:
                    areas = sorted([b[4] for b in boxes], reverse=True)
                    if areas[2] < areas[0] / 3.0:
                        continue

                accepted = True
                best_rgb_anom = rgb_anom
                best_mask = mask
                best_boxes = boxes
                break

            if not accepted:
                try:
                    rgb_anom, mask = dent_mod.process_image(
                        scene_dir=item.scene_dir,
                        image_id=item.image_id,
                        rng=rng,
                        objects_per_image=2,
                        dents_per_object=1,
                        mode="box_crumple",
                    )
                except Exception:
                    continue
                boxes = contours_boxes(mask)[:2]
                best_rgb_anom, best_mask, best_boxes = rgb_anom, mask, boxes

            rgb_soft, depth_heat = build_soft_rgb_and_depth_heat(rgb, depth, best_mask, best_rgb_anom)
            h, w = rgb_soft.shape[:2]
            yolo_lines = boxes_to_yolo(best_boxes, width=w, height=h)

            stem = f"{item.scene_id}_{item.image_id}"
            cv2.imwrite(str(rgb_ds / split_name / "images" / f"{stem}.png"), rgb_soft)
            (rgb_ds / split_name / "labels" / f"{stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

            cv2.imwrite(str(depth_ds / split_name / "images" / f"{stem}.png"), depth_heat)
            (depth_ds / split_name / "labels" / f"{stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

    write_data_yaml(rgb_ds)
    write_data_yaml(depth_ds)

    print(f"RGB dataset: {rgb_ds}")
    print(f"Depth dataset: {depth_ds}")
    print(
        f"Split sizes: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dent-module-path", type=Path, default=Path(__file__).with_name("apply_realistic_dents.py"))
    parser.add_argument("--seed", type=int, default=177)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-retries", type=int, default=40)
    args = parser.parse_args()

    build_datasets(
        scenes_root=args.scenes_root,
        output_root=args.output_root,
        dent_module_path=args.dent_module_path,
        seed=args.seed,
        overwrite=args.overwrite,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
