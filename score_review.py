"""
Score a filled-in review sheet to estimate how the detector did on real field images.

Reads field_data/camera_XX/review/review_sheet.csv after a human has filled in
the human_trash_count and verdict columns, then reports precision and recall
estimates with 95 percent confidence intervals.

Usage:
    python score_review.py 11

Verdict column must be one of:
    correct      model and human agree (trash present in positive, none in negative)
    false_pos    model boxed something that is not trash
    false_neg    model missed trash that is there

Author: Griffin Willer
"""

import csv
import math
import os
import sys

OUTPUT_ROOT = "field_data"


def wilson(k, n, z=1.96):
    # 95 percent confidence interval for a proportion
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0, center - half), min(1, center + half)


def main():
    if len(sys.argv) < 2:
        print("Usage: python score_review.py <camera_id>")
        return
    cam = int(sys.argv[1])
    cam_dir = os.path.join(OUTPUT_ROOT, "camera_%02d" % cam)
    sheet = os.path.join(cam_dir, "review", "review_sheet.csv")
    summary = os.path.join(cam_dir, "summary.txt")

    with open(sheet) as f:
        rows = list(csv.DictReader(f))

    pos = [r for r in rows if r["stratum"] == "positive"]
    neg = [r for r in rows if r["stratum"] == "negative"]
    unfilled = [r for r in rows if not r["verdict"].strip()]
    if unfilled:
        print("%d rows still have no verdict. Fill them in first." % len(unfilled))
        return

    # Precision from the positive sample
    pos_correct = sum(1 for r in pos if r["verdict"].strip() == "correct")
    p, p_lo, p_hi = wilson(pos_correct, len(pos))

    # Miss rate from the negative sample
    neg_missed = sum(1 for r in neg if r["verdict"].strip() == "false_neg")
    m, m_lo, m_hi = wilson(neg_missed, len(neg))

    # Scale to the whole card to get recall
    n_pos_total = n_neg_total = None
    with open(summary) as f:
        for line in f:
            if line.startswith("Images processed:"):
                n_img = int(line.split(":")[1])
            if line.startswith("Images with trash:"):
                n_pos_total = int(line.split(":")[1].split("(")[0])
    n_neg_total = n_img - n_pos_total

    est_tp = p * n_pos_total
    est_fn = m * n_neg_total
    recall = est_tp / (est_tp + est_fn) if (est_tp + est_fn) > 0 else 0.0
    recall_lo = (p_lo * n_pos_total) / (p_lo * n_pos_total + m_hi * n_neg_total + 1e-9)
    recall_hi = (p_hi * n_pos_total) / (p_hi * n_pos_total + m_lo * n_neg_total + 1e-9)

    print()
    print("Camera %d field validation" % cam)
    print("-" * 44)
    print("Positive sample reviewed: %d" % len(pos))
    print("  confirmed trash:        %d" % pos_correct)
    print("  false alarms:           %d" % (len(pos) - pos_correct))
    print("Negative sample reviewed: %d" % len(neg))
    print("  trash the model missed: %d" % neg_missed)
    print()
    print("Image level precision:  %.1f%%  (95%% CI %.1f to %.1f)" % (100 * p, 100 * p_lo, 100 * p_hi))
    print("Miss rate on negatives: %.1f%%  (95%% CI %.1f to %.1f)" % (100 * m, 100 * m_lo, 100 * m_hi))
    print("Image level recall:     %.1f%%  (approx CI %.1f to %.1f)" % (100 * recall, 100 * recall_lo, 100 * recall_hi))
    print()
    print("Whole card: %d images, %d flagged by model, %d not flagged" % (n_img, n_pos_total, n_neg_total))
    print("Estimated true trash images on card: %.0f" % (est_tp + est_fn))

    out = os.path.join(cam_dir, "review", "validation_result.txt")
    with open(out, "w") as f:
        f.write("camera,%d\n" % cam)
        f.write("positive_sample,%d\n" % len(pos))
        f.write("positive_confirmed,%d\n" % pos_correct)
        f.write("negative_sample,%d\n" % len(neg))
        f.write("negative_missed,%d\n" % neg_missed)
        f.write("precision,%.4f,%.4f,%.4f\n" % (p, p_lo, p_hi))
        f.write("miss_rate,%.4f,%.4f,%.4f\n" % (m, m_lo, m_hi))
        f.write("recall,%.4f,%.4f,%.4f\n" % (recall, recall_lo, recall_hi))
    print("Saved", out)


if __name__ == "__main__":
    main()
