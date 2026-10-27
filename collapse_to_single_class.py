"""
Collapse an existing multi-class YOLO label set to single-class.
Rewrites the class id (first number on each line) to 0 for every .txt.
Operates on train/ and val/ label folders in place.

SAFE: only touches the first token per line. Coordinates are untouched.
Run this ONCE on your combined_dataset labels after backing them up.
"""
import os, glob

# EDIT these two paths to point at your label folders:
LABEL_DIRS = [
    "mega_training/combined_dataset/train/labels",
    "mega_training/combined_dataset/val/labels",
]

def collapse(label_dir):
    files = glob.glob(os.path.join(label_dir, "*.txt"))
    changed = 0
    for path in files:
        with open(path) as f:
            lines = f.read().splitlines()
        new_lines = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            parts[0] = "0"                 # force class id to 0
            new_lines.append(" ".join(parts))
        with open(path, "w") as f:
            f.write("\n".join(new_lines))
        changed += 1
    print(f"{label_dir}: rewrote {changed} label files to class 0")

if __name__ == "__main__":
    for d in LABEL_DIRS:
        if os.path.isdir(d):
            collapse(d)
        else:
            print(f"SKIP (not found): {d}")
