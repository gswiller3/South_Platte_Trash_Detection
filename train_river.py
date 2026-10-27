"""
Fine tune the single class trash model on a river background set.

Starts from a base model, freezes the backbone, and trains the detection head for
a few epochs at a low learning rate so the model learns that river texture is
background without forgetting what trash looks like.

Usage:
    python train_river.py
    python train_river.py --data river_finetune_v7/river_finetune.yaml --name river_v7 --device 0

Device: mps for Apple silicon, 0 for an NVIDIA GPU, cpu if you have neither.

Author: Griffin Willer
"""

import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="best.pt", help="model to start from")
ap.add_argument("--data", default="river_finetune/river_finetune.yaml", help="dataset yaml from build_river_finetune.py")
ap.add_argument("--name", default="river_v6", help="run name, change it for each new run")
ap.add_argument("--device", default="mps", help="mps, 0, or cpu")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--imgsz", type=int, default=640)
args = ap.parse_args()

model = YOLO(args.base)
model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=16,
    device=args.device,
    freeze=10,          # keep the backbone, train neck and head
    lr0=0.001,
    lrf=0.1,
    warmup_epochs=1,
    patience=6,
    close_mosaic=5,
    project="runs/detect",
    name=args.name,
    exist_ok=True,
    seed=7,
)

print()
print("Done. New model at runs/detect/runs/detect/%s/weights/best.pt" % args.name)
print("Copy it next to the scripts, for example: cp runs/detect/runs/detect/%s/weights/best.pt best_river.pt" % args.name)
