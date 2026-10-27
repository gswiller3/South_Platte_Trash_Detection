"""
Build a small fine tune dataset that teaches the model what empty river looks like.

Takes clean daylight frames off a camera card as background images (empty labels)
and mixes them with a sample of the original trash training set so the model
keeps its trash knowledge while learning that water texture is not trash.

Usage:
    python build_river_finetune.py /Volumes/SD_10 --camera 10
    python build_river_finetune.py /Volumes/SD_07 --camera 7 --trash-dir river_finetune_v6 --out river_finetune_v7
    python build_river_finetune.py /Volumes/SD_07 --camera 7 --trash-dir river_finetune_v6 --out river_tiles2 --tile 2

The trash images can come from combined_dataset or from an earlier river_finetune set,
anything laid out as <dir>/train/images + labels and <dir>/val/images + labels.
Frames with empty label files are skipped as trash sources, so river backgrounds in an
old set are not reused by accident.

--tile N cuts every river frame into the same N by N overlapping grid that
process_camera.py --tiles N uses at detection time, and saves each tile as its own
background image. Train on that set and run the card with --tiles N and the model
sees ripples at the same size in training and in the field. Without --tile, whole
frames are saved and the card is run without --tiles. Do not mix the two.

Author: Griffin Willer
"""

import argparse
import os
import random
import shutil

import cv2

ORIGINAL_DATASET = "combined_dataset"     # default trash source, override with --trash-dir
OUT_DIR = "river_finetune"                # default output, override with --out
N_RIVER_TRAIN = 800        # empty river frames for training
N_RIVER_VAL = 200          # empty river frames for validation
N_TRASH_TRAIN = 2500       # original trash images for training
N_TRASH_VAL = 400          # original trash images for validation
MAX_SIDE = 1280            # shrink huge trail cam frames so training is not slow
SKIP_FOLDERS = ("100MEDIA",)   # first folder has camera setup frames, skip it
NIGHT_CHANNEL_DIFF = 3.0
RANDOM_SEED = 7
TILE_OVERLAP = 0.15        # must match TILE_OVERLAP in process_camera.py

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def tile_grid(w, h, n, overlap=TILE_OVERLAP):
    # Same grid as process_camera.py, kept identical on purpose.
    if n <= 1:
        return [(0, 0, w, h)]
    tw, th = w / float(n), h / float(n)
    ox, oy = int(tw * overlap), int(th * overlap)
    wins = []
    for r in range(n):
        for c in range(n):
            x1 = max(0, int(c * tw) - ox)
            y1 = max(0, int(r * th) - oy)
            x2 = min(w, int((c + 1) * tw) + ox)
            y2 = min(h, int((r + 1) * th) + oy)
            wins.append((x1, y1, x2, y2))
    return wins


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
    ap.add_argument("--trash-dir", default=ORIGINAL_DATASET, help="YOLO folder with trash images and labels to mix in")
    ap.add_argument("--out", default=OUT_DIR, help="output folder for the new fine tune set")
    ap.add_argument("--tile", type=int, default=1, help="cut each river frame into an N by N grid, to match process_camera.py --tiles N")
    args = ap.parse_args()
    random.seed(RANDOM_SEED)
    out_dir = args.out
    tiles = max(1, args.tile)
    per_frame = tiles * tiles
    # With tiling, each frame gives several background images, so fewer frames are
    # needed to keep the background count near 1,000 against 2,900 trash images.
    n_train_frames = -(-N_RIVER_TRAIN // per_frame)
    n_val_frames = -(-N_RIVER_VAL // per_frame)
    if os.path.abspath(args.trash_dir) == os.path.abspath(out_dir):
        print("trash-dir and out must be different folders")
        return
    if not os.path.isdir(os.path.join(args.trash_dir, "train", "images")):
        print("No trash images found at %s/train/images" % args.trash_dir)
        print("Download river_finetune_v6.zip from the GitHub release and pass --trash-dir river_finetune_v6")
        return

    dirs = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = os.path.join(out_dir, split, kind)
            os.makedirs(d, exist_ok=True)
            dirs[(split, kind)] = d

    # Step 1: river backgrounds
    cands = card_images(args.card)
    random.shuffle(cands)
    print("Card has %d candidate frames (setup folder skipped)" % len(cands))
    if tiles > 1:
        print("Tiling %d by %d, so %d frames give %d background images" % (tiles, tiles, n_train_frames + n_val_frames, (n_train_frames + n_val_frames) * per_frame))
    need = n_train_frames + n_val_frames
    kept = []
    for p in cands:
        if len(kept) >= need:
            break
        img = cv2.imread(p)
        if img is None or is_night_frame(img):
            continue
        kept.append((p, img))
        if len(kept) % 100 == 0:
            print("  %d daylight frames collected" % len(kept))
    print("Collected %d daylight river frames" % len(kept))

    n_bg = {"train": 0, "val": 0}
    for i, (p, img) in enumerate(kept):
        split = "train" if i < n_train_frames else "val"
        h, w = img.shape[:2]
        for t, (x1, y1, x2, y2) in enumerate(tile_grid(w, h, tiles)):
            name = "river_cam%02d_%05d" % (args.camera, i)
            if tiles > 1:
                name += "_t%d" % t
            cv2.imwrite(os.path.join(dirs[(split, "images")], name + ".jpg"), shrink(img[y1:y2, x1:x2]),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            open(os.path.join(dirs[(split, "labels")], name + ".txt"), "w").close()
            n_bg[split] += 1
    print("Wrote %d train and %d val background images" % (n_bg["train"], n_bg["val"]))

    # Step 2: sample of the original trash set
    def sample_original(split, n):
        src_img = os.path.join(args.trash_dir, split, "images")
        src_lbl = os.path.join(args.trash_dir, split, "labels")
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
    yaml_path = os.path.join(out_dir, "river_finetune.yaml")
    with open(yaml_path, "w") as f:
        f.write("# built by build_river_finetune.py, tile: %d\n" % tiles)
        f.write("path: %s\n" % os.path.abspath(out_dir))
        f.write("train: train/images\nval: val/images\n\nnc: 1\nnames:\n  0: trash\n")
    print()
    print("Wrote", yaml_path)
    print("Train images:", len(os.listdir(dirs[("train", "images")])))
    print("Val images:  ", len(os.listdir(dirs[("val", "images")])))
    print()
    print("Next: python3 train_river.py --data %s --name <run name>" % yaml_path)
    if tiles > 1:
        print("Then run cards with: python3 process_camera.py <card> <camera> --tiles %d --model <new model>" % tiles)


if __name__ == "__main__":
    main()
