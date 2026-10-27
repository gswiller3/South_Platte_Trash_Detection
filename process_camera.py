"""
Process one trail camera SD card through the trash detector.

Copies the images off the card, runs the model, writes a detections CSV,
saves annotated copies of every image with trash, picks a showcase set,
and builds random review folders for manual validation.

Usage:
    python process_camera.py /Volumes/CARDNAME 11
    python process_camera.py ~/Desktop/some_folder 11 --review-n 50

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
MODEL_PATH = "best.pt"
OUTPUT_ROOT = "field_data"
CONF_THRESHOLD = 0.25
IMAGE_SIZE = 1280            # larger than training size so small trash is not lost in big frames
LOOKBACK_FRAMES = 5          # how many previous frames to compare for repeat trash
REPEAT_IOU = 0.5             # overlap above this means same piece of trash as before
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


def draw_boxes(img, boxes, confs, new_flags):
    for box, conf, is_new in zip(boxes, confs, new_flags):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 255, 255) if is_new else (0, 165, 255)   # yellow new, orange repeat
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        tag = "trash %.2f" % conf + ("" if is_new else " (repeat)")
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
    ap.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    ap.add_argument("--review-n", type=int, default=REVIEW_N)
    ap.add_argument("--skip-copy", action="store_true", help="source is already a local copy")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images, for a quick test")
    ap.add_argument("--skip-first", type=int, default=0, help="drop the first N images, for camera setup frames")
    ap.add_argument("--daylight-only", action="store_true", help="skip grayscale IR night frames")
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
        for p in images:
            dst = os.path.join(raw_dir, os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copy2(p, dst)
            local.append(dst)
    print("  %d images ready" % len(local))

    # Step 2: order by time so repeat detection works
    print("Reading timestamps...")
    records = [(get_timestamp(p), p) for p in local]
    records.sort()

    # Step 3: run the model
    print("Loading model", args.model)
    model = YOLO(args.model)
    print("  model classes:", model.names)

    rows = []
    recent_boxes = []     # list of box lists from the last few frames
    positives = []        # (max_conf, filename) for images with trash
    negatives = []        # filenames with no trash
    night_skipped = []    # filenames judged to be IR night frames
    total_boxes = total_new = 0

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
                "confidence": "", "is_new": "",
            })
            if (i + 1) % 100 == 0:
                print("  %d / %d done" % (i + 1, len(records)))
            continue

        res = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes, confs = [], []
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy().tolist()
            confs = res.boxes.conf.cpu().numpy().tolist()

        # Repeat check: same spot as any box in the last few frames
        new_flags = []
        for b in boxes:
            seen = any(iou(b, pb) > REPEAT_IOU for frame in recent_boxes for pb in frame)
            new_flags.append(not seen)
        recent_boxes.append(boxes)
        if len(recent_boxes) > LOOKBACK_FRAMES:
            recent_boxes.pop(0)

        n_boxes = len(boxes)
        n_new = sum(new_flags)
        total_boxes += n_boxes
        total_new += n_new
        max_conf = max(confs) if confs else 0.0

        if n_boxes:
            positives.append((max_conf, fname))
            out = draw_boxes(img, boxes, confs, new_flags)
            cv2.imwrite(os.path.join(ann_dir, fname), out)
        else:
            negatives.append(fname)

        for j, (b, c, nf) in enumerate(zip(boxes, confs, new_flags)):
            rows.append({
                "camera_id": args.camera_id,
                "filename": fname,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "box_index": j,
                "x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3]),
                "confidence": round(c, 3),
                "is_new": int(nf),
            })
        if not boxes:
            rows.append({
                "camera_id": args.camera_id, "filename": fname,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "box_index": -1, "x1": "", "y1": "", "x2": "", "y2": "",
                "confidence": "", "is_new": "",
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
        "New trash (repeats removed): %d" % total_new,
        "Review set: %d positives, %d negatives" % (len(pos_sample), len(neg_sample)),
        "Confidence threshold: %.2f   Inference size: %d" % (args.conf, args.imgsz),
    ]
    with open(os.path.join(cam_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print()
    print("\n".join(summary))
    print()
    print("Outputs in", cam_dir)


if __name__ == "__main__":
    main()
