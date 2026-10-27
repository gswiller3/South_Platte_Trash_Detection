"""
Build a small fine tune dataset that teaches the model what empty river looks like.

Takes clean daylight frames off a camera card as background images (empty labels)
and mixes them with a sample of the original trash training set so the model
keeps its trash knowledge while learning that water texture is not trash.

Usage:
    python build_river_finetune.py /Volumes/SD_10

Author: Griffin Willer
"""

import argparse
import os
import random
import shutil

import cv2

ORIGINAL_DATASET = "combined_dataset"     # the single class set from the Group 7 share
OUT_DIR = "river_finetune"
N_RIVER_TRAIN = 800        # empty river frames for training
N_RIVER_VAL = 200          # empty river frames for validation
N_TRASH_TRAIN = 2500       # original trash images for training
N_TRASH_VAL = 400          # original trash images for validation
MAX_SIDE = 1280            # shrink huge trail cam frames so training is not slow
SKIP_FOLDERS = ("100MEDIA",)   # first folder has camera setup frames, skip it
NIGHT_CHANNEL_DIFF = 3.0
RANDOM_SEED = 7

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def is_night_frame(img):
    h, w = img.shape[:2]
    crop = img[int(h * 0.1):int(h * 0.85), int(w * 0.1):int(w * 0.9)]
    small = cv2.resize(crop, (256, 256))
    b, g, r = cv2.split(small)
    diff = (cv2.absdiff(b, g).mean() + cv2.absdiff(g, r).mean()) / 2.0
    return diff < NIGHT_CHANNEL_DIFF


def shrink(img):
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def card_images(root):
    out = []
    for dirpath, _, files in os.walk(root):
        if os.path.basename(dirpath) in SKIP_FOLDERS:
            continue
        for f in files:
            if f.startswith(".") or not f.lower().endswith(IMAGE_EXTS):
                continue
            out.append(os.path.join(dirpath, f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("card", help="SD card path or folder of river frames")
    ap.add_argument("--camera", type=int, default=10, help="camera number, used in filenames")
    args = ap.parse_args()
    random.seed(RANDOM_SEED)

    dirs = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = os.path.join(OUT_DIR, split, kind)
            os.makedirs(d, exist_ok=True)
            dirs[(split, kind)] = d

    # Step 1: river backgrounds
    cands = card_images(args.card)
    random.shuffle(cands)
    print("Card has %d candidate frames (setup folder skipped)" % len(cands))
    need = N_RIVER_TRAIN + N_RIVER_VAL
    kept = []
    for p in cands:
        if len(kept) >= need:
            break
        img = cv2.imread(p)
        if img is None or is_night_frame(img):
            continue
        kept.append((p, shrink(img)))
        if len(kept) % 100 == 0:
            print("  %d daylight frames collected" % len(kept))
    print("Collected %d daylight river frames" % len(kept))

    for i, (p, img) in enumerate(kept):
        split = "train" if i < N_RIVER_TRAIN else "val"
        name = "river_cam%02d_%05d" % (args.camera, i)
        cv2.imwrite(os.path.join(dirs[(split, "images")], name + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(os.path.join(dirs[(split, "labels")], name + ".txt"), "w").close()

    # Step 2: sample of the original trash set
    def sample_original(split, n):
        src_img = os.path.join(ORIGINAL_DATASET, split, "images")
        src_lbl = os.path.join(ORIGINAL_DATASET, split, "labels")
        files = [f for f in os.listdir(src_img) if f.lower().endswith(IMAGE_EXTS)]
        random.shuffle(files)
        copied = 0
        for f in files:
            stem = os.path.splitext(f)[0]
            lbl = os.path.join(src_lbl, stem + ".txt")
            if not os.path.exists(lbl) or os.path.getsize(lbl) == 0:
                continue   # only keep images that actually have trash boxes
            shutil.copy2(os.path.join(src_img, f), os.path.join(dirs[(split, "images")], f))
            shutil.copy2(lbl, os.path.join(dirs[(split, "labels")], stem + ".txt"))
            copied += 1
            if copied >= n:
                break
        print("Copied %d original trash images into %s" % (copied, split))

    sample_original("train", N_TRASH_TRAIN)
    sample_original("val", N_TRASH_VAL)

    # Step 3: dataset yaml
    yaml_path = os.path.join(OUT_DIR, "river_finetune.yaml")
    with open(yaml_path, "w") as f:
        f.write("path: %s\n" % os.path.abspath(OUT_DIR))
        f.write("train: train/images\nval: val/images\n\nnc: 1\nnames:\n  0: trash\n")
    print()
    print("Wrote", yaml_path)
    print("Train images:", len(os.listdir(dirs[("train", "images")])))
    print("Val images:  ", len(os.listdir(dirs[("val", "images")])))
    print()
    print("Next: python3 train_river.py")


if __name__ == "__main__":
    main()
