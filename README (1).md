# Aerial Trash Detection — Group 7 CV Final Project

A YOLOv8-based system that detects and counts illegally dumped trash from aerial drone
footage. The detector is trained as a **single class** ("trash"), since for this project we
only care whether something is trash and how much of it there is, not the specific type.

Current best model: **~97% mAP50** on the aerial validation set (precision ~98%, recall ~93%),
trained for 40 epochs on a combined dataset that includes the UAVVaste aerial litter images.

---

## What's in this folder

| File / folder | What it is |
|---|---|
| `best.pt` | **The trained model.** This is the current best detector. Use this to run detection. |
| `trash_detector.py` | Runs the model on a folder of images and saves annotated results. |
| `train_mega.py` | Training script. Only needed if you want to train a new model. |
| `trash_single.yaml` | Dataset config for training (points at the images/labels, sets 1 class). |
| `collapse_to_single_class.py` | Converts multi-class labels to single class (class 0). |
| `convert_uavvaste_singleclass.py` | Converts UAVVaste COCO annotations to single-class YOLO labels. |
| `combined_dataset/` | The training images and labels (large). Only needed for training. |

---

## Quick start: run detection with the trained model

You do **not** need to train anything to use the model. Just use `best.pt`.

1. Install the requirements (one time):

   ```
   pip install ultralytics opencv-python numpy
   ```

2. Run the detector on a folder of images:

   ```
   python trash_detector.py ./input_images ./output_results --model ./best.pt
   ```

   Replace `./input_images` with the folder of images you want to run on. Annotated
   results (boxes + confidence) are saved to `./output_results`, and the script prints how
   many detections it found in each image.

That's it. `best.pt` is the trained model, and the script loads it and runs detection. Every
box is labeled using the model's own class name, which for our single-class model is "trash."

---

## How the model and the code relate (important)

Training does **not** change the code. The `.py` scripts are the instructions; training
produces a separate output file, `best.pt`, which holds the learned model. Think of the
code as the recipe and `best.pt` as the result.

So:
- To **use** the current model, you only need `best.pt` (plus `trash_detector.py` to run it).
- To **make a new model**, you run the training script, which produces a **new** `best.pt`.
- The code stays the same; the `best.pt` is the thing that gets better as we train.

---

## Doing additional training (optional)

Only do this if you want to improve the model. You'll need the `combined_dataset/` folder.

1. Make sure `trash_single.yaml` points at the dataset location on your machine. Open it and
   set the `path:` line to wherever `combined_dataset` lives for you.

2. Start training (fine-tunes from the current best model rather than from scratch):

   ```
   python train_mega.py
   ```

   Or run it directly with Ultralytics, starting from the current best weights:

   ```
   caffeinate -i python3 -c "
   from ultralytics import YOLO
   model = YOLO('best.pt')
   model.train(data='trash_single.yaml', epochs=50, imgsz=640,
               batch=16, device='mps', project='runs', name='trash_new')
   "
   ```

   (`caffeinate -i` keeps a Mac awake during the long run. On a non-Mac, drop it and set
   `device=0` for an NVIDIA GPU or `device='cpu'` if you have no GPU.)

3. When training finishes, your new model is saved at:

   ```
   runs/detect/trash_new/weights/best.pt
   ```

   The training also writes `results.csv` in that same run folder, which has the final
   metrics (look at the last row: `mAP50(B)`, `precision(B)`, `recall(B)`).

---

## Keeping everyone on the newest best model

There is no automatic syncing. Each person who trains produces their own `best.pt`. To stay
coordinated:

1. Whenever someone trains a model that **beats the current best** (higher mAP50), rename it
   with the score and date, for example `best_mAP98_0813.pt`, and upload it to our shared
   folder.
2. Post the new numbers to the group so everyone knows there's a better model.
3. Everyone else grabs the new `best.pt` from the shared folder when they need it.

Rule of thumb: **code stays in sync through our shared folder; the `best.pt` is what we swap
out as we get better results.**

---

## Notes

- The dataset is large (many GB), so it lives in the shared drive folder, not in the code repo.
- If you'd rather rebuild the dataset from scratch instead of downloading it, the UAVVaste
  images come from the public UAVVaste dataset, and `convert_uavvaste_singleclass.py` plus
  `collapse_to_single_class.py` turn everything into the single-class format used here.
- Common gotcha: run scripts from inside the folder that contains your images/model, or use
  full paths, otherwise Python won't find the files.
