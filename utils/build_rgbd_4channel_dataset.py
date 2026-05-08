import argparse
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def fuse_pair(rgb_path, depth_path, out_path):
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    dtm = cv2.imread(str(depth_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    if dtm is None:
        raise FileNotFoundError(depth_path)
    if rgb.shape[:2] != dtm.shape[:2]:
        dtm = cv2.resize(dtm, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    depth_1ch = cv2.cvtColor(dtm, cv2.COLOR_BGR2GRAY)
    fused = np.concatenate([rgb, depth_1ch[..., None]], axis=2).astype(np.uint8)
    fused = np.ascontiguousarray(fused)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]                      
    if cv2.imwrite(str(out_path), fused, png_params) is not True:
                                                                                     
        try:
            from PIL import Image
        except ImportError as ie:
            raise OSError(f"imwrite failed: {out_path} (нет Pillow: {ie})") from ie
        rgba = np.dstack([fused[..., 2], fused[..., 1], fused[..., 0], fused[..., 3]])
        Image.fromarray(rgba.astype(np.uint8), "RGBA").save(str(out_path), compress_level=0)


def write_data_yaml(ds_root, nc, names):
    lines = [
        f"channels: 4",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {nc}",
        f"names: {names!r}".replace('"', "'"),
    ]
    (ds_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def build_fused_rgbd(rgb_root, depth_root, output_root, *, overwrite):
    rgb_root = rgb_root.expanduser().resolve()
    depth_root = depth_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if output_root.exists() and overwrite:
                                                                                                       
        if os.name == "nt":
            shutil.rmtree(output_root)
        else:
            subprocess.run(
                ["/bin/bash", "-c", f"chmod -Rf u+w '{output_root}' 2>/dev/null; rm -rf '{output_root}'"],
                check=True,
            )

    splits = {"train": "train", "valid": "valid", "test": "test"}
    n_ok = 0
    for split in splits.values():
        r_img = rgb_root / split / "images"
        d_img = depth_root / split / "images"
        r_lbl = rgb_root / split / "labels"
        out_i = output_root / split / "images"
        out_l = output_root / split / "labels"
        if not r_img.is_dir():
            raise FileNotFoundError(r_img)

        stems = sorted(p.stem for p in r_img.glob("*.png"))
        dstems = {p.stem for p in d_img.glob("*.png")}
        missing = [s for s in stems if s not in dstems]
        if missing:
            raise RuntimeError(f"Missing depth images for split {split} (examples): {missing[:5]}")

        for stem in stems:
            fuse_pair(r_img / f"{stem}.png", d_img / f"{stem}.png", out_i / f"{stem}.png")
            src_lbl = r_lbl / f"{stem}.txt"
            dst_lbl = out_l / f"{stem}.txt"
            out_l.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_lbl, dst_lbl)
            n_ok += 1
            if n_ok % 500 == 0:
                print(f"... fused {n_ok} ...", flush=True)

    write_data_yaml(output_root, nc=1, names=["defect"])
    print(f"Built {output_root}: {n_ok} fused PNGs + paired labels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-root", type=Path, required=True)
    ap.add_argument("--depth-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    build_fused_rgbd(args.rgb_root, args.depth_root, args.output_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
