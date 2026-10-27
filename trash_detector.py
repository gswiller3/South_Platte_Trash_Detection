"""
Single-class trash detector.

Runs the trained YOLOv8 model on a folder of images and saves annotated
copies with bounding boxes and confidence scores. The model is single class,
so every detection is labeled using the model's own class name ("trash").

Usage:
    python trash_detector.py ./input_images ./output_results --model ./best.pt

Author: Griffin Willer
"""

import argparse
import os
import cv2
from ultralytics import YOLO

# Settings
CONFIDENCE_THRESHOLD = 0.25   # ignore detections below this confidence
IMAGE_SIZE = 640              # inference size (match training)
BOX_COLOR = (0, 255, 0)       # green boxes
HIGH_CONF_COLOR = (0, 255, 255)  # yellow for confident detections
BOX_THICKNESS = 2
FONT_SCALE = 0.5

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def main():
    parser = argparse.ArgumentParser(description="Single-class trash detector")
    parser.add_argument("input_dir", help="Folder of images to run on")
    parser.add_argument("output_dir", help="Folder to save annotated results")
    parser.add_argument("--model", default="best.pt", help="Path to the trained model (best.pt)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    args = parser.parse_args()

    # Load the trained model
    print("Loading model:", args.model)
    model = YOLO(args.model)

    # Gather input images
    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(IMAGE_EXTS)]
    if not files:
        print("No images found in", args.input_dir)
        return
    os.makedirs(args.output_dir, exist_ok=True)
    print("Found", len(files), "images. Running detection...")

    total_detections = 0
    for fname in files:
        path = os.path.join(args.input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print("  could not read", fname)
            continue

        # Run the model on this image
        results = model.predict(img, conf=args.conf, imgsz=IMAGE_SIZE, verbose=False)[0]

        count = 0
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = box.astype(int)
                # Use the model's own class name (single class -> "trash")
                class_name = model.names[cls_id]
                color = HIGH_CONF_COLOR if conf > 0.5 else BOX_COLOR
                cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                label = "%s %.2f" % (class_name, conf)
                cv2.putText(img, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 2)
                cv2.putText(img, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, 1)
                count += 1

        total_detections += count
        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, img)
        print("  %s: %d detections" % (fname, count))

    print("Done. %d images processed, %d total detections. Results in %s"
          % (len(files), total_detections, args.output_dir))


if __name__ == "__main__":
    main()
