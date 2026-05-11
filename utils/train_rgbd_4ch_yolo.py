from __future__ import annotations

import argparse
import sys
from pathlib import Path


def apply_ultralytics_4channel_loader_patch():
    import cv2

    from ultralytics.data import base as data_base

    if getattr(data_base.BaseDataset.__init__, "_rgbd4ch_patch", False):
        return

    orig_init = data_base.BaseDataset.__init__

    def wrapped_init(self, *args, channels=3, **kwargs):
        orig_init(self, *args, channels=channels, **kwargs)

        if getattr(self, "channels", 3) not in (1, 3):
            self.cv2_flag = cv2.IMREAD_UNCHANGED

    wrapped_init._rgbd4ch_patch = True
    data_base.BaseDataset.__init__ = wrapped_init


def parse_args(argv: list[str]):
    rep = Path(__file__).resolve().parents[1]
    default_data = Path.home() / "Документы" / "Датасеты" / "defects_rgbd_4ch_det" / "data.yaml"
    default_project = rep / "training_runs" / "defects_rgbd_4ch_det"

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(default_data))
    ap.add_argument("--model", type=str, default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--project", type=str, default=str(default_project))
    ap.add_argument("--name", type=str, default="yolo26n_rgbd4ch_smoke")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--verify-loader-only", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args(argv)


def verify_four_channel_dataloader(data_yaml: str, imgsz: int):
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    cfg_over = dict(data=data_yaml, imgsz=imgsz, task="detect", fraction=1.0, rect=False)
    cfg = get_cfg(overrides=cfg_over)
    data = check_det_dataset(data_yaml, autodownload=False)
    if int(data.get("channels", 0)) != 4:
        raise SystemExit(f"data.yaml должен содержать channels = 4, сейчас: {data.get('channels')}")
    tp = data["train"]
    if isinstance(tp, list):
        tp = tp[0]
    stride = 32
    ds = build_yolo_dataset(cfg, tp, batch=1, data=data, mode="train", stride=stride, rect=False)
    img = ds[0]["img"]
    shape = getattr(img, "shape", ())
    print(f"[verify-loader] первый образ: shape={shape}")
    if len(shape) != 3 or int(shape[0]) != 4:
        raise SystemExit(f"[verify-loader] FAIL: ожидалось (4, H, W), получили {shape}")
    print("[verify-loader] OK - 4 входных канала в батче.")


def main(argv: list[str] | None = None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.verify_loader_only:
        apply_ultralytics_4channel_loader_patch()
        verify_four_channel_dataloader(args.data, args.imgsz)
        return

    from ultralytics import YOLO

    pretrained = not args.no_pretrained

    print(f"[rgbd-train] YOLO load: {args.model}", flush=True)
    model = YOLO(args.model)
    apply_ultralytics_4channel_loader_patch()
    print("[rgbd-train] model.train(...) start", flush=True)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        pretrained=pretrained,
        plots=args.plots,
        exist_ok=True,
        verbose=True,
        fraction=args.fraction,
        val=not args.no_val,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
