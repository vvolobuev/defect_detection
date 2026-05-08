from pathlib import Path
import argparse

import cv2
import numpy as np


def color_for_idx(idx: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(idx + 123)
    c = rng.integers(40, 255, size=3, endpoint=False)
    return int(c[0]), int(c[1]), int(c[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--image-id", type=str, default="000001")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = args.scene_dir.parent / "result_images" / f"objects_contours_{args.image_id}.png"

    rgb_path = args.scene_dir / "rgb" / f"{args.image_id}.png"
    mask_dir = args.scene_dir / "mask_visib"

    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")

    masks = sorted(mask_dir.glob(f"{args.image_id}_*.png"))
    if not masks:
        raise FileNotFoundError(f"No masks found for image id {args.image_id} in {mask_dir}")

    vis = image.copy()
    for m in masks:
        obj_token = m.stem.split("_")[1]
        obj_idx = int(obj_token)

        mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        color = color_for_idx(obj_idx)
        cv2.drawContours(vis, contours, -1, color, 2)

                                             
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(
            vis,
            f"id:{obj_idx}",
            (x, max(18, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), vis)
    print(f"Saved: {args.out}")
    print(f"Objects visualized: {len(masks)}")


if __name__ == "__main__":
    main()
