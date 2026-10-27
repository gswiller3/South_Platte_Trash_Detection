"""
Process one trail camera SD card through the trash detector.

Copies the images off the card, runs the model, writes a detections CSV,
saves annotated copies of every image with trash, picks a showcase set,
and builds random review folders for manual validation.

Usage:
    python process_camera.py /Volumes/CARDNAME 11
    python process_camera.py ~/Desktop/some_folder 11 --review-n 50
    python process_camera.py /Volumes/CARDNAME 11 --tiles 2 --model best_river_tiles2.pt

Scale: the model was trained on whole frames at 640 pixels, so detection runs at 640
on the whole frame by default and the two match. If small trash is being missed,
--tiles 2 cuts each frame into a 2 by 2 grid of overlapping tiles and runs each tile
at 640, which makes everything in the frame about twice as big to the model. A tiled
run needs a model trained on tiles, see build_river_finetune.py --tile.

Author: Griffin Willer
"""

import argparse
import csv
import os
import random
import shutil
from datetime import datetime

import cv2
from PIL import Image
from ultralytics import YOLO

# Settings
MODEL_PATH = "best_river.pt"  # the river fine tuned model, best.pt alone boxes ripples
OUTPUT_ROOT = "field_data"
CONF_THRESHOLD = 0.25
IMAGE_SIZE = 640             # same as training, so ripples look the same size they did in training
TILES = 1                    # 1 = whole frame. 2 = 2 by 2 grid of tiles, each run at IMAGE_SIZE. Needs a model trained with --tile 2
TILE_OVERLAP = 0.15          # fraction of a tile shared with its neighbour so trash on a seam is whole in at least one tile
MERGE_IOU = 0.5              # boxes from two tiles that overlap more than this are the same item
LOOKBACK_FRAMES = 30         # previous frames checked for stationary repeats, 30 = 5 min at 10 s interval
REPEAT_IOU = 0.5             # overlap above this means same piece of trash sitting in the same spot
DRIFT_RADIUS = 0.35          # fraction of image diagonal, box center within this of a box in the previous frame = same item drifting
DRIFT_MAX_GAP_S = 25         # previous frame must be this recent in seconds for the drift match to count
SHOWCASE_N = 10              # top images by confidence copied to showcase folder
REVIEW_N = 50                # random images per stratum for manual review
RANDOM_SEED = 42             # fixed so the review sample can be reproduced
NIGHT_CHANNEL_DIFF = 3.0     # below this the frame is grayscale IR, treated as night

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VIDEO_EXTS = (".mp4", ".avi", ".mov")


def find_images(root):
    images, videos = [], []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.startswith("._") or f.startswith("."):
                continue
            low = f.lower()
            if low.endswith(IMAGE_EXTS):
                images.append(os.path.join(dirpath, f))
            elif low.endswith(VIDEO_EXTS):
                videos.append(os.path.join(dirpath, f))
    return sorted(images), sorted(videos)


def get_timestamp(path):
    # EXIF first, file time as a fallback
    try:
        exif = Image.open(path)._getexif()
        if exif and 36867 in exif:
            return datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.fromtimestamp(os.path.getmtime(path))


def is_night_frame(img):
    # IR night shots are grayscale, so the color channels nearly match.
    # Check the middle of the frame and skip the camera's info bar at the bottom.
    h, w = img.shape[:2]
    crop = img[int(h * 0.1):int(h * 0.85), int(w * 0.1):int(w * 0.9)]
    small = cv2.resize(crop, (256, 256))
    b, g, r = cv2.split(small)
    diff = (cv2.absdiff(b, g).mean() + cv2.absdiff(g, r).mean()) / 2.0
    return diff < NIGHT_CHANNEL_DIFF


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def tile_grid(w, h, n, overlap=TILE_OVERLAP):
    # Pixel windows (x1, y1, x2, y2) that cut a w by h frame into an n by n grid.
    # Each window is widened by overlap times the tile size on its inner sides so an
    # item sitting on a seam is whole in at least one tile. n = 1 is the whole frame.
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


def merge_boxes(boxes, confs):
    # An item in the overlap strip is seen by two tiles. Keep the higher confidence
    # box and drop any box that mostly covers the same spot. Two boxes are the same
    # spot if their overlap is above MERGE_IOU, or if one is mostly inside the other,
    # which is what a half item on a seam looks like next to the whole item.
    order = sorted(range(len(boxes)), key=lambda k: confs[k], reverse=True)
    keep = []
    for k in order:
        same = False
        for j in keep:
            a, b = boxes[k], boxes[j]
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if iou(a, b) > MERGE_IOU or inter / (smaller + 1e-9) > 0.7:
                same = True
                break
        if not same:
            keep.append(k)
    return [boxes[k] for k in keep], [confs[k] for k in keep]


def detect(model, img, conf, imgsz, tiles):
    # Runs the model on the whole frame, or on each tile of an n by n grid, and
    # returns boxes in full frame pixel coordinates with their confidences.
    h, w = img.shape[:2]
    wins = tile_grid(w, h, tiles)
    crops = [img[y1:y2, x1:x2] for (x1, y1, x2, y2) in wins]
    results = model.predict(crops, conf=conf, imgsz=imgsz, verbose=False)
    boxes, confs = [], []
    for (x1, y1, _, _), res in zip(wins, results):
        if res.boxes is None or not len(res.boxes):
            continue
        for b, c in zip(res.boxes.xyxy.cpu().numpy().tolist(), res.boxes.conf.cpu().numpy().tolist()):
            boxes.append([b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1])
            confs.append(float(c))
    if tiles > 1 and boxes:
        boxes, confs = merge_boxes(boxes, confs)
    return boxes, confs


def classify_repeats(boxes, recent, prev_boxes, prev_dt_s, diag):
    # Returns one label per box: new, stationary, or drift.
    # stationary: overlaps a box from any of the last LOOKBACK_FRAMES frames (snagged item)
    # drift: center is near an unmatched box from the immediately previous frame (item floating through)
    labels = []
    used_prev = set()
    for b in boxes:
        if any(iou(b, pb) > REPEAT_IOU for frame in recent for pb in frame):
            labels.append("stationary")
            continue
        label = "new"
        if prev_boxes and prev_dt_s is not None and prev_dt_s <= DRIFT_MAX_GAP_S:
            cx, cy = center(b)
            best, best_d = None, DRIFT_RADIUS * diag
            for k, pb in enumerate(prev_boxes):
                if k in used_prev:
                    continue
                px, py = center(pb)
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = k, d
            if best is not None:
                used_prev.add(best)
                label = "drift"
        labels.append(label)
    return labels


def draw_boxes(img, boxes, confs, labels):
    for box, conf, label in zip(boxes, confs, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        is_new = label == "new"
        color = (0, 255, 255) if is_new else (0, 165, 255)   # yellow new, orange repeat
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        tag = "trash %.2f" % conf + ("" if is_new else " (%s)" % label)
        cv2.putText(img, tag, (x1 + 4, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
        cv2.putText(img, tag, (x1 + 4, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="SD card path or folder of images")
    ap.add_argument("camera_id", type=int, help="camera number 1 to 12")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="inference size, keep it equal to the training size")
    ap.add_argument("--tiles", type=int, default=TILES, help="1 = whole frame, 2 = 2 by 2 grid of tiles, needs a model trained with --tile 2")
    ap.add_argument("--review-n", type=int, default=REVIEW_N)
    ap.add_argument("--skip-copy", action="store_true", help="source is already a local copy")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images, for a quick test")
    ap.add_argument("--skip-first", type=int, default=0, help="drop the first N images, for camera setup frames")
    ap.add_argument("--daylight-only", action="store_true", help="skip grayscale IR night frames")
    ap.add_argument("--lookback", type=int, default=LOOKBACK_FRAMES, help="frames checked for stationary repeats")
    args = ap.parse_args()

    cam = "camera_%02d" % args.camera_id
    cam_dir = os.path.join(OUTPUT_ROOT, cam)
    raw_dir = os.path.join(cam_dir, "raw")
    ann_dir = os.path.join(cam_dir, "annotated")
    show_dir = os.path.join(cam_dir, "showcase")
    rev_pos = os.path.join(cam_dir, "review", "positives")
    rev_neg = os.path.join(cam_dir, "review", "negatives")
    for d in (raw_dir, ann_dir, show_dir, rev_pos, rev_neg):
        os.makedirs(d, exist_ok=True)

    # Step 1: copy images off the card
    images, videos = find_images(args.source)
    if not images:
        print("No images found under", args.source)
        return
    print("Found %d images and %d videos on %s" % (len(images), len(videos), args.source))
    if videos:
        print("  Videos are skipped. Tell Griff how many there were: %d" % len(videos))
    if args.skip_first:
        images = images[args.skip_first:]
        print("  --skip-first set, dropped first %d images (setup frames)" % args.skip_first)
    if args.limit:
        images = images[:args.limit]
        print("  --limit set, using first %d images only" % len(images))

    if args.skip_copy:
        local = images
    else:
        print("Copying to", raw_dir)
        local = []
        taken = {}            # copy name -> source path, so two DCIM folders with the same file names do not collide
        for p in images:
            name = os.path.basename(p)
            if name in taken and taken[name] != p:
                name = os.path.basename(os.path.dirname(p)) + "_" + name
            taken[name] = p
            dst = os.path.join(raw_dir, name)
            if not os.path.exists(dst):
                shutil.copy2(p, dst)
            local.append(dst)
    print("  %d images ready" % len(local))

    # Step 2: order by time so repeat detection works
    print("Reading timestamps...")
    records = [(get_timestamp(p), p) for p in local]
    records.sort()

    # Step 3: run the model
    if not os.path.exists(args.model):
        print("Model file not found:", args.model)
        print("Download it from https://github.com/gswiller3/South_Platte_Trash_Detection/releases/tag/v6-river")
        print("and put it in the same folder as this script, or pass --model with the path to it.")
        return
    print("Loading model", args.model)
    model = YOLO(args.model)
    print("  model classes:", model.names)

    rows = []
    recent_boxes = []     # list of box lists from the last LOOKBACK_FRAMES frames
    prev_boxes = []       # boxes from the immediately previous processed frame
    prev_ts = None
    positives = []        # (max_conf, filename) for images with trash
    negatives = []        # filenames with no trash
    night_skipped = []    # filenames judged to be IR night frames
    total_boxes = total_new = total_stationary = total_drift = 0

    print("Running detection on %d images..." % len(local))
    for i, (ts, path) in enumerate(records):
        fname = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print("  could not read", fname)
            continue

        if args.daylight_only and is_night_frame(img):
            night_skipped.append(fname)
            rows.append({
                "camera_id": args.camera_id, "filename": fname,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "box_index": -2, "x1": "", "y1": "", "x2": "", "y2": "",
                "confidence": "", "is_new": "", "repeat_type": "",
            })
            if (i + 1) % 100 == 0:
                print("  %d / %d done" % (i + 1, len(records)))
            continue

        boxes, confs = detect(model, img, args.conf, args.imgsz, args.tiles)

        # Repeat check: stationary (same spot in recent frames) or drift (moved from the previous frame)
        h, w = img.shape[:2]
        diag = (h * h + w * w) ** 0.5
        prev_dt = (ts - prev_ts).total_seconds() if prev_ts is not None else None
        labels = classify_repeats(boxes, recent_boxes, prev_boxes, prev_dt, diag)
        recent_boxes.append(boxes)
        if len(recent_boxes) > args.lookback:
            recent_boxes.pop(0)
        prev_boxes, prev_ts = boxes, ts

        n_boxes = len(boxes)
        n_new = labels.count("new")
        total_boxes += n_boxes
        total_new += n_new
        total_stationary += labels.count("stationary")
        total_drift += labels.count("drift")
        max_conf = max(confs) if confs else 0.0

        if n_boxes:
            positives.append((max_conf, fname))
            out = draw_boxes(img, boxes, confs, labels)
            cv2.imwrite(os.path.join(ann_dir, fname), out)
        else:
            negatives.append(fname)

        for j, (b, c, lab) in enumerate(zip(boxes, confs, labels)):
            rows.append({
                "camera_id": args.camera_id,
                "filename": fname,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "box_index": j,
                "x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3]),
                "confidence": round(c, 3),
                "is_new": int(lab == "new"),
                "repeat_type": lab,
            })
        if not boxes:
            rows.append({
                "camera_id": args.camera_id, "filename": fname,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "box_index": -1, "x1": "", "y1": "", "x2": "", "y2": "",
                "confidence": "", "is_new": "", "repeat_type": "",
            })

        if (i + 1) % 100 == 0:
            print("  %d / %d done" % (i + 1, len(records)))

    # Step 4: write detections csv
    csv_path = os.path.join(cam_dir, "detections.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Step 5: showcase = highest confidence images
    positives.sort(reverse=True)
    for conf, fname in positives[:SHOWCASE_N]:
        shutil.copy2(os.path.join(ann_dir, fname), os.path.join(show_dir, fname))

    # Step 6: random review sets
    random.seed(RANDOM_SEED)
    pos_names = [f for _, f in positives]
    pos_sample = random.sample(pos_names, min(args.review_n, len(pos_names)))
    neg_sample = random.sample(negatives, min(args.review_n, len(negatives)))
    for fname in pos_sample:
        shutil.copy2(os.path.join(ann_dir, fname), os.path.join(rev_pos, fname))
    for fname in neg_sample:
        shutil.copy2(os.path.join(raw_dir, fname), os.path.join(rev_neg, fname))

    sheet = os.path.join(cam_dir, "review", "review_sheet.csv")
    with open(sheet, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera_id", "filename", "stratum", "model_boxes",
                    "human_trash_count", "verdict", "notes"])
        box_count = {}
        for r in rows:
            if r["box_index"] != -1:
                box_count[r["filename"]] = box_count.get(r["filename"], 0) + 1
        for fname in pos_sample:
            w.writerow([args.camera_id, fname, "positive", box_count.get(fname, 0), "", "", ""])
        for fname in neg_sample:
            w.writerow([args.camera_id, fname, "negative", 0, "", "", ""])

    # Step 7: summary
    n_img = len(records) - len(night_skipped)
    dates = [r[0] for r in records]
    summary = [
        "Camera %d summary" % args.camera_id,
        "Source: %s" % args.source,
        "Date range: %s to %s" % (min(dates).strftime("%Y-%m-%d %H:%M"), max(dates).strftime("%Y-%m-%d %H:%M")),
        "Images on card: %d" % len(records),
        "Night frames skipped: %d" % len(night_skipped),
        "Setup frames skipped: %d" % args.skip_first,
        "Images processed: %d" % n_img,
        "Videos skipped: %d" % len(videos),
        "Images with trash: %d (%.1f%%)" % (len(positives), 100.0 * len(positives) / max(n_img, 1)),
        "Total trash boxes: %d" % total_boxes,
        "Stationary repeats (same spot, last %d frames): %d" % (args.lookback, total_stationary),
        "Drift repeats (same item moved from previous frame): %d" % total_drift,
        "New trash (repeats removed): %d" % total_new,
        "Review set: %d positives, %d negatives" % (len(pos_sample), len(neg_sample)),
        "Confidence threshold: %.2f   Inference size: %d   Tiles: %d   Model: %s" % (args.conf, args.imgsz, args.tiles, args.model),
    ]
    with open(os.path.join(cam_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print()
    print("\n".join(summary))
    print()
    print("Outputs in", cam_dir)


if __name__ == "__main__":
    main()
