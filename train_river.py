"""
Fine tune the single class trash model on the river background set.

Starts from best.pt, freezes the backbone, and trains the detection head for
a few epochs at a low learning rate so the model learns that river texture
is background without forgetting what trash looks like.

Usage:
    python train_river.py

Author: Griffin Willer
"""

from ultralytics import YOLO

START_MODEL = "best.pt"
DATA = "river_finetune/river_finetune.yaml"
RUN_NAME = "river_v6"

model = YOLO(START_MODEL)
model.train(
    data=DATA,
    epochs=15,
    imgsz=640,
    batch=16,
    device="mps",
    freeze=10,          # keep the backbone, train neck and head
    lr0=0.001,
    lrf=0.1,
    warmup_epochs=1,
    patience=6,
    close_mosaic=5,
    project="runs/detect",
    name=RUN_NAME,
    exist_ok=True,
    seed=7,
)

print()
print("Done. New model at runs/detect/%s/weights/best.pt" % RUN_NAME)
print("Copy it next to the scripts as best_river.pt and rerun the card with --model best_river.pt")
