import argparse
from pathlib import Path

import cv2
import numpy as np


def load_object_masks(mask_visib_dir, image_id):
    masks = []
    for path in sorted(mask_visib_dir.glob(f"{image_id}_*.png")):
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 0).astype(np.uint8)
        if int(m.sum()) > 250:
            masks.append(m)
    return masks


def make_object_dent_field(obj_mask, rng, dents_per_object):
    h, w = obj_mask.shape
    ys, xs = np.where(obj_mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    box_w = max(8, x1 - x0 + 1)
    box_h = max(8, y1 - y0 + 1)

    dist = cv2.distanceTransform((obj_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
    valid = dist > 4.0
    vy, vx = np.where(valid)
    if len(vx) == 0:
        return np.zeros((h, w), dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.zeros((h, w), dtype=np.float32)

                                                   
    idx = int(rng.integers(0, len(vx)))
    cx = float(vx[idx])
    cy = float(vy[idx])

    obj_area = float(obj_mask.sum())
                                                                   
    target_area = rng.uniform(1.0 / 6.0, 1.0 / 3.0) * obj_area

    aspect = rng.uniform(0.72, 1.38)
    rx = np.sqrt((target_area * aspect) / np.pi)
    ry = np.sqrt(target_area / (np.pi * aspect))
    rx = float(np.clip(rx, 4.0, 0.45 * box_w))
    ry = float(np.clip(ry, 4.0, 0.45 * box_h))
    angle = rng.uniform(0.0, np.pi)

    x = xx - cx
    y = yy - cy
    ca, sa = np.cos(angle), np.sin(angle)
    xr = ca * x + sa * y
    yr = -sa * x + ca * y

    g = np.exp(-0.5 * ((xr / rx) ** 2 + (yr / ry) ** 2))
    strength = rng.uniform(0.45, 0.90)
    field -= strength * g

    ring = np.exp(-0.5 * (((xr / (rx * 1.25)) ** 2 + (yr / (ry * 1.25)) ** 2)))
    field += 0.05 * strength * np.maximum(ring - g, 0.0)

    field *= obj_mask.astype(np.float32)
    return cv2.GaussianBlur(field, (0, 0), 1.6)


def make_crumple_field(obj_mask, rng, folds_per_object):
    h, w = obj_mask.shape
    ys, xs = np.where(obj_mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    box_w = max(8, x1 - x0 + 1)
    box_h = max(8, y1 - y0 + 1)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.zeros((h, w), dtype=np.float32)
    dist = cv2.distanceTransform((obj_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
    valid = dist > 3.0
    vy, vx = np.where(valid)
    if len(vx) == 0:
        return field

    for _ in range(folds_per_object):
        idx = int(rng.integers(0, len(vx)))
        cx, cy = float(vx[idx]), float(vy[idx])
        angle = rng.uniform(0.0, np.pi)
        nx, ny = np.cos(angle), np.sin(angle)

                                      
        d = (xx - cx) * nx + (yy - cy) * ny
        width = rng.uniform(0.030, 0.075) * max(box_w, box_h)
        depth = rng.uniform(0.42, 0.80)
        fold = -depth * np.exp(-np.abs(d) / (width + 1e-6))

                                                  
        tx, ty = -ny, nx
        t = (xx - cx) * tx + (yy - cy) * ty
        extent = rng.uniform(0.20, 0.42) * max(box_w, box_h)
        window = np.exp(-0.5 * (t / (extent + 1e-6)) ** 2)
        field += fold * window

                                                                                          
        rr = ((xx - cx) / (0.18 * box_w + 1e-6)) ** 2 + ((yy - cy) / (0.18 * box_h + 1e-6)) ** 2
        bowl = np.exp(-0.5 * rr)
        field -= rng.uniform(0.10, 0.22) * bowl

    field *= obj_mask.astype(np.float32)
    return cv2.GaussianBlur(field, (0, 0), 1.8)


def apply_field_to_rgb(image, field):
    h, w = field.shape
    dent_strength = np.maximum(-field, 0.0)

                                   
    gy, gx = np.gradient(dent_strength)
    shift_scale = min(h, w) * 0.14
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = map_x + gx.astype(np.float32) * shift_scale
    map_y = map_y + gy.astype(np.float32) * shift_scale
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

                                                              
    nxx, nyy = np.gradient(field)
    nx = -nxx
    ny = -nyy
    nz = np.ones_like(field, dtype=np.float32)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    light = np.array([0.48, -0.32, 0.82], dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
    shade = (shade - shade.mean()) * 0.30

                                                                   
    out = warped.astype(np.float32) / 255.0
    d = np.clip(dent_strength, 0.0, 1.0)
    depth_dark = np.clip(1.0 - 0.58 * d, 0.35, 1.0)                                     

                                                 
    shape = np.clip(1.0 + 0.24 * shade, 0.84, 1.16)
    out *= (depth_dark * shape)[:, :, None]
    out = np.clip(out, 0.0, 1.0)

                                                            
    alpha = cv2.GaussianBlur(dent_strength, (0, 0), 2.4)
    alpha = np.clip(alpha * 2.1, 0.0, 1.0)[:, :, None]
    base = image.astype(np.float32) / 255.0
    out = base * (1.0 - alpha) + out * alpha
    out = np.clip(out, 0.0, 1.0)

                                                                
    dent_mask = (dent_strength > 0.10).astype(np.uint8) * 255
    return (out * 255).astype(np.uint8), dent_mask


def process_image(scene_dir, image_id, rng, objects_per_image, dents_per_object, mode):
    rgb_path = scene_dir / "rgb" / f"{image_id}.png"
    mask_visib_dir = scene_dir / "mask_visib"

    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"RGB not found: {rgb_path}")

    obj_masks = load_object_masks(mask_visib_dir, image_id)
    if not obj_masks:
        raise RuntimeError(f"No object masks found for image id {image_id}")

    pick_n = min(objects_per_image, len(obj_masks))
                                                                           
    areas = np.array([int(m.sum()) for m in obj_masks], dtype=np.int64)
    idxs = np.argsort(areas)[-pick_n:]

    h, w = image.shape[:2]
    full_field = np.zeros((h, w), dtype=np.float32)
    for order, i in enumerate(idxs):
                                                                                
        if mode == "box_crumple":
            full_field += make_crumple_field(obj_masks[int(i)], rng, folds_per_object=1)
        elif mode == "hybrid":
            if order % 2 == 0:
                full_field += make_object_dent_field(obj_masks[int(i)], rng, dents_per_object=1)
            else:
                full_field += make_crumple_field(obj_masks[int(i)], rng, folds_per_object=1)
        else:
            full_field += make_object_dent_field(obj_masks[int(i)], rng, dents_per_object=1)

    return apply_field_to_rgb(image, full_field)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--image-ids", nargs="+", default=["000001", "000007"])
    parser.add_argument("--objects-per-image", type=int, default=2)
    parser.add_argument("--dents-per-object", type=int, default=1)
    parser.add_argument("--mode", choices=["dent", "box_crumple", "hybrid"], default="dent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.scene_dir.parent / "result_images" / "rgb_dented"
    if args.mask_dir is None:
        args.mask_dir = args.scene_dir.parent / "result_images" / "dent_mask"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    processed = 0
    for image_id in args.image_ids:
        out, mask = process_image(
            scene_dir=args.scene_dir,
            image_id=image_id,
            rng=rng,
            objects_per_image=args.objects_per_image,
            dents_per_object=args.dents_per_object,
            mode=args.mode,
        )
        cv2.imwrite(str(args.output_dir / f"{image_id}.png"), out)
        cv2.imwrite(str(args.mask_dir / f"{image_id}.png"), mask)
        processed += 1

    print(f"Processed: {processed} images")
    print(f"Output dir: {args.output_dir}")
    print(f"Mask dir: {args.mask_dir}")


if __name__ == "__main__":
    main()
