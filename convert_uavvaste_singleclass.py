"""
Convert UAVVaste COCO annotations to YOLO SINGLE-CLASS format.
Every rubbish box becomes class 0 = "trash".
"""
import json, os

COCO_JSON = "UAVVasteDataset/annotations/annotations.json"
OUT_LABELS_DIR = "UAVVasteDataset/labels"
TARGET_CLASS_ID = 0   # single class: everything is "trash"

def convert():
    with open(COCO_JSON) as f:
        coco = json.load(f)
    images = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in coco["images"]}
    per_image = {}
    for ann in coco["annotations"]:
        per_image.setdefault(ann["image_id"], []).append(ann)
    os.makedirs(OUT_LABELS_DIR, exist_ok=True)
    written, skipped = 0, 0
    for image_id, (file_name, W, H) in images.items():
        lines = []
        for ann in per_image.get(image_id, []):
            x, y, w, h = ann["bbox"]
            xc, yc, wn, hn = (x + w/2)/W, (y + h/2)/H, w/W, h/H
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
                skipped += 1; continue
            lines.append(f"{TARGET_CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        stem = os.path.splitext(file_name)[0]
        with open(os.path.join(OUT_LABELS_DIR, stem + ".txt"), "w") as out:
            out.write("\n".join(lines))
        written += 1
    print(f"UAVVaste: wrote {written} single-class label files, skipped {skipped} bad boxes")

if __name__ == "__main__":
    convert()
