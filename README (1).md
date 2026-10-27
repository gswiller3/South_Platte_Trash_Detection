# South Platte River Trash Detection

Colorado School of Mines senior capstone with Denver Trout Unlimited. Trail cameras under 11 bridges on the South Platte between Evans Ave and York St photograph the water every 10 seconds. A YOLOv8 model counts the floating trash in those frames so the client can see where trash enters the river and build a heat map by bridge.

Lead CV developer: Griffin Willer. Fall 2026.

---

## The models

Two trained weight files. Both are too big to commit to the repo, so they are attached to the v6-river release: https://github.com/gswiller3/South_Platte_Trash_Detection/releases/tag/v6-river. Download them into the same folder as the scripts.

| File | What it is | Validation numbers | Use it for |
|---|---|---|---|
| `best.pt` | Single class YOLOv8m, 155 MB. Trained on about 43,500 images: TACO, TrashNet, Kaggle waste sets and UAVVaste aerial litter, all collapsed to one class called `trash`. 40 epochs. | mAP50 97.4, precision 98.4, recall 93.4 | Starting point for any new fine tune. Do not run it on river frames by itself. |
| `best_river.pt` | The river model. `best.pt` fine tuned for 15 epochs on 1,000 empty South Platte frames plus 2,900 trash images, backbone frozen, learning rate 0.001. 52 MB. | mAP50 97.5, precision 97.9, recall 91.4 | **This is the model to run on camera cards.** It is the default in `process_camera.py`. |

### Why single class

The earlier 20 class model topped out at 71.8 percent mAP50 because it spent its capacity telling bottles from bags from wrappers. For a heat map we only need to know that trash is present and roughly how much. Collapsing to one class took the aerial validation score to 97 percent. Bridge cameras look straight down at the water, which is the same geometry as the drone imagery in UAVVaste, so that dataset carried over well.

### Why the river fine tune

`best.pt` had never seen moving water. On the first SD card it boxed moonlight reflections at night and ripples during the day, on frames with no trash in them at all. The fix was background training: `build_river_finetune.py` pulls 1,000 daylight frames off a camera card, gives each an empty label file, mixes in a sample of the original trash images so the model does not forget what trash looks like, and `train_river.py` retrains from `best.pt` with the backbone frozen so only the detection head moves. Took 45 minutes on an M4 Max. Validation precision held at 98 percent with the river frames in the val set, which means the model learned that water texture is background.

### What this repo lets you reproduce

- **Run detection on camera cards.** Fully. Code here, weights on the release.
- **Fine tune on new river frames.** Fully. The fine tune set used for `best_river.pt` is on the release as `river_finetune_v6.zip`, and `build_river_finetune.py` can grow it from any new card.
- **Rebuild `best.pt` from scratch.** Not from this repo alone. That took about 43,500 images pulled from public datasets and a 42 hour training run. The sources and scripts are listed at the bottom under Rebuilding the base model, but nobody needs to do that to use or improve the system.

### Scale, and why detection runs at 640

The model was trained on whole frames shrunk to 640 pixels on the long side, so `process_camera.py` runs detection at 640 on the whole frame. The two have to match. An earlier version ran detection at 1280, and on the first card the river model boxed ripples in 99 of 200 daylight frames, worse than the base model, because the ripples showed up at twice the size it had learned to ignore. Leave `--imgsz` at 640 unless the model is retrained at a different size.

The cost of 640 on a 4608 pixel wide frame is that a 10 cm item seen from a typical bridge height is roughly 10 pixels across, which is near the limit of what the model can pick up. If the manual review shows small items being missed, use tiles. `--tiles 2` cuts each frame into a 2 by 2 grid of overlapping tiles, runs each tile at 640 and merges the boxes back into frame coordinates, so everything is about twice as big to the model. It is four model passes per frame, so about four times slower. A tiled run needs a model trained on tiles, which is one extra flag in the retraining steps below. Never run a tiled model on whole frames or a whole frame model on tiles, the ripples would be at the wrong size again.

---

## Setting this up on your own computer

You need Python, the code from this repo, and the model weights from the Releases page. Nothing else. No GPU required, a GPU just makes it faster.

**1. Python.** Version 3.10 or newer. Check with `python3 --version`. If that fails on Windows try `python --version`. Install from python.org if you have neither.

**2. Get the code.** Either click the green Code button on this page, Download ZIP, and unzip it somewhere sensible, or if you have git:

```
git clone https://github.com/gswiller3/South_Platte_Trash_Detection.git
cd South_Platte_Trash_Detection
```

**3. Install the libraries.** From inside that folder:

```
pip3 install -r requirements.txt
```

On Windows use `pip` instead of `pip3`. This pulls in ultralytics, which pulls in PyTorch, so it is a few hundred MB and takes a few minutes. If pip complains about an externally managed environment on a Mac, add `--break-system-packages` to the end, or make a virtual environment first:

```
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**4. Get the model.** Go to https://github.com/gswiller3/South_Platte_Trash_Detection/releases/tag/v6-river, scroll to Assets, download `best_river.pt` and put it in the same folder as the scripts. Download `best.pt` and `river_finetune_v6.zip` too if you plan to retrain. Confirm it loaded:

```
python3 -c "from ultralytics import YOLO; print(YOLO('best_river.pt').names)"
```

You want `{0: 'trash'}`. The first run of ultralytics may print a one time setup message, that is normal.

**5. Test on a handful of images.** Make a folder with a few JPGs in it, any photos of trash on the ground will do, then:

```
python3 trash_detector.py ./test_images ./test_out --model best_river.pt
```

Open `test_out` and you should see your images with boxes drawn. If that works, everything is installed.

**6. Hardware notes.**

- Apple silicon Mac: the scripts use the GPU automatically through MPS, and `train_river.py` defaults to `--device mps`. A 200 frame test took about 3 minutes at 1280 including the copy off the card, and 640 is lighter than that.
- Windows or Linux with an NVIDIA card: install the CUDA build of PyTorch first, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`, then pass `--device 0` to `train_river.py`. Detection runs on the GPU without changes.
- No GPU: everything still runs, detection is a few seconds per frame, so a full 9,000 frame card is an overnight job. Pass `--device cpu` to `train_river.py` if you train, and expect training to take a day instead of an hour.
- The SD card path is `/Volumes/CARDNAME` on a Mac, `E:\` or similar on Windows, `/media/username/CARDNAME` on Linux. The scripts take any folder path, so you can also copy the DCIM folder to your hard drive first and point at that.
- `caffeinate -i` in the commands below is Mac only. Leave it off on Windows and Linux and just stop the machine from sleeping in settings.

**7. Folder layout once you are set up.** Everything in one folder:

```
South_Platte_Trash_Detection/
    best_river.pt               from the release
    best.pt                     from the release, only if retraining
    river_finetune_v6/          from the release zip, only if retraining
    process_camera.py
    score_review.py
    build_field_workbook.py
    trash_detector.py
    build_river_finetune.py
    train_river.py
    requirements.txt
    South_Platte_Predicted_Trash_Dataset.xlsx   optional, gives the results workbook its predicted columns
    field_data/                 created by process_camera.py
    South_Platte_Field_Results.xlsx             created by build_field_workbook.py
    runs/                       created by training
```

---

## Running a camera card

Plug in the card, find its name with `ls /Volumes`, then:

```
python3 process_camera.py /Volumes/SD_10 10 --limit 200 --daylight-only
```

That is a 200 frame test. Look at `field_data/camera_10/showcase` and make sure the boxes are on trash. Then the full card:

```
rm -rf field_data/camera_10
caffeinate -i python3 process_camera.py /Volumes/SD_10 10 --daylight-only --skip-first 60
```

The camera number is the second argument. It goes into the folder name and every row of the detections file, so double check it against the label on the card. `--skip-first` drops the frames of you installing the camera. `--daylight-only` skips the grayscale night frames. `caffeinate -i` keeps a Mac awake. The defaults are `best_river.pt` at 640 on the whole frame; `--model`, `--imgsz`, `--conf` and `--tiles` override them, and the settings used are written into `summary.txt`.

Output lands in `field_data/camera_XX/`:

| Path | What it is |
|---|---|
| `raw/` | Untouched copies off the card |
| `annotated/` | Every frame that had a detection, boxes drawn. Yellow is new, orange is a repeat. |
| `showcase/` | Top 10 by confidence, for the client |
| `detections.csv` | One row per box: camera, filename, timestamp, coordinates, confidence, repeat type |
| `review/positives/` and `review/negatives/` | Random samples for the manual check |
| `review/review_sheet.csv` | Fill in the verdict column by hand |
| `summary.txt` | The numbers for this camera |

### How repeats are handled

A piece of trash shows up in more than one frame. The script sorts frames by timestamp and tags every box:

- `stationary`: overlaps a box from any of the last 30 frames, 5 minutes at 10 seconds. Trash snagged on a rock. Counted once.
- `drift`: center is near an unmatched box in the previous frame and that frame is less than 25 seconds old. The same item floating through. Counted once on the frame it first appeared.
- `new`: neither. This is the count.

`summary.txt` reports total boxes, stationary repeats, drift repeats, and new trash. Report the last one. Known gaps: an item that drops out of detection for more than 5 minutes gets counted again, two items close together in one frame can merge, and an item that circulates in an eddy and comes back gets counted again. All small at low flow.

### Validating the count

Open `review/review_sheet.csv` and the two review folders. For each positive, mark `correct` or `false_pos`. For each negative, mark `correct` or `false_neg`. Save, then:

```
python3 score_review.py 10
```

Prints precision, miss rate and estimated recall with 95 percent confidence intervals using the Wilson method. Fifty per stratum gives about plus or minus 14 points. `--review-n 200` on the processing run tightens it to about 7. Do the full review on three or four cameras and apply those rates to the rest with a note. No count goes to the client without this step; the workbook below carries the precision next to the count so the two travel together.

### Building the results workbook

Once a card or two has been through the steps above:

```
python3 build_field_workbook.py
```

It walks `field_data/`, reads every `camera_XX` folder, and writes `South_Platte_Field_Results.xlsx`. Nothing is typed in by hand, so run it again after every card and the workbook is rebuilt from scratch. Sheets:

| Sheet | What it holds |
|---|---|
| `Cameras` | One row per camera: frames, night and setup frames skipped, hours of daylight covered, images with trash, total boxes, stationary and drift repeats, new items, items per daylight hour, precision and recall if the review has been scored, and the model and settings used |
| `By_Period` | Camera by 14 day period, P1 Sep 28 to Oct 11, P2 Oct 12 to Oct 25, P3 Oct 26 to Nov 8. New items, frames, hours, rate per hour, raw boxes, days with frames |
| `Comparison` | Same layout as the Comparison sheet in `South_Platte_Predicted_Trash_Dataset.xlsx`: predicted low, mid and high next to the observed count and rate for every camera and period, with the actual cells filled by the script. The predicted columns are read from that workbook if it is in the same folder, otherwise from `heatmap_periods_long.csv`, otherwise left blank |
| `Heatmap_Long` | One row per camera per period with lat, lon, observed items, rate and a 0 to 1 weight. Also written as `heatmap_observed_long.csv` for kepler.gl or QGIS |
| `Daily` | One row per camera per day with frames, hours, new items, the predicted number and the Denver gauge flow for that day. Also written as `daily_observed_long.csv` |
| `Validation` | Precision, miss rate and recall with intervals for every camera that has a scored review sheet |
| `Detections` | Every box from every camera with timestamp, period, coordinates, confidence and repeat type, with a filter on the header row |

The camera clocks were set a year behind, so the timestamps on the cards say 2025. The script rewrites the year to 2026 before sorting frames into periods and the `Cameras` sheet says how many timestamps it fixed. Hours covered are frames times the 10 second interval, so compare cameras on items per daylight hour rather than raw counts; no two cards cover the same hours. Report new items, not total boxes.

---

## Retraining with more river frames

Every card processed is more background data, and this is the one training step anyone can do from the repo. You need `best.pt` and `river_finetune_v6.zip` from the release, unzipped next to the scripts.

**1. Build a new fine tune set.** Point the builder at a new card and at the previous set as the source of trash images:

```
python3 build_river_finetune.py /Volumes/SD_07 --camera 7 --trash-dir river_finetune_v6 --out river_finetune_v7
```

It pulls 800 daylight river frames for training and 200 for validation off the card, writes an empty label file for each so the model learns them as background, and copies 2,500 train and 400 val trash images from the previous set. Night frames are skipped by checking whether the color channels match, which is what a grayscale IR shot looks like. The first DCIM folder on the card is skipped because it is setup frames. River frames from the old set are not reused, only its trash images, so each new set has fresh backgrounds.

Want frames from several cameras in one set? Copy the DCIM folders from each card into one folder and point the builder at that.

**2. Train.**

```
caffeinate -i python3 train_river.py --data river_finetune_v7/river_finetune.yaml --name river_v7 --device mps
```

Use `--device 0` for an NVIDIA GPU or `--device cpu` with no GPU. About 45 minutes on an M4 Max, 15 epochs, backbone frozen. The last lines print the validation numbers. You want precision and mAP50 to stay in the high 90s and recall above 90.

**3. Swap in the new model.**

```
cp runs/detect/runs/detect/river_v7/weights/best.pt best_river.pt
```

Then rerun a 200 frame test on a card and look at the showcase folder. If the ripple boxes are down and real trash still gets boxed, keep it. If not, go back to the old one and change something.

**4. Tiles, only if small items are being missed.** Build the set with `--tile 2`, which cuts every river frame into the same 2 by 2 overlapping grid that `process_camera.py --tiles 2` uses, so the model trains on river texture at the size it will see it. Fewer frames are pulled off the card so the background count stays near 1,000 against 2,900 trash images.

```
python3 build_river_finetune.py /Volumes/SD_07 --camera 7 --trash-dir river_finetune_v6 --out river_tiles2 --tile 2
caffeinate -i python3 train_river.py --data river_tiles2/river_finetune.yaml --name river_tiles2
cp runs/detect/runs/detect/river_tiles2/weights/best.pt best_river_tiles2.pt
python3 process_camera.py /Volumes/SD_10 10 --limit 200 --daylight-only --tiles 2 --model best_river_tiles2.pt
```

Keep the tiled model and the whole frame model as separate files and always pair `--tiles 2` with the tiled one. The `Tiles` line in `summary.txt` records which was used.

---

## Files

| File | What it does |
|---|---|
| `process_camera.py` | Runs a card through the model, writes everything above |
| `score_review.py` | Scores the manual review sheet |
| `build_field_workbook.py` | Builds `South_Platte_Field_Results.xlsx` from every camera folder in `field_data/` |
| `build_river_finetune.py` | Builds the river background training set from a card, whole frames or tiles |
| `train_river.py` | Fine tunes `best.pt` on that set |
| `trash_detector.py` | Simple detector, one folder in, annotated folder out, no repeat logic. Fine for a quick look. |
| `train_mega.py` | Original training script for `best.pt`, see Rebuilding the base model |
| `trash_single.yaml` | Dataset config, one class |
| `collapse_to_single_class.py` | Rewrites multi class labels to class 0 |
| `convert_uavvaste_singleclass.py` | Converts UAVVaste COCO annotations to YOLO single class |
| `requirements.txt` | The Python libraries, `pip3 install -r requirements.txt` |
| `build_prediction_dataset.py` and `sources_list.py` | Builds the predicted counts per bridge per period from gauge data, rain and litter studies |
| `FIELD_DATA_INSTRUCTIONS.md` | Step by step for processing a card, with the questions to report back |
| `HEATMAP_PRACTICE_README.md` | How to load the predicted dataset into a heat map tool |

Not in the repo: `best.pt`, `best_river.pt` and `river_finetune_v6.zip` are on the release. `field_data/` and `runs/` are generated locally. The full `combined_dataset` is not shared, see below.

---

## Gotchas

- The camera clocks were set a year wrong. Timestamps say 2025. Frame order is still right, which is all the repeat logic needs.
- HEIC cannot be read by OpenCV. The GardePro cards are JPG so this only matters for phone photos.
- Keep the scripts, the model and the yaml in the same folder, or use full paths.
- Do not run `trash_detector_v3.py` or `quickstart.py` from the old repo. They use the 20 class model and label everything as plastic_bag.
- If the Mac fans go full speed and progress stops, the GPU is choking on the image size. The default 640 whole frame is light, so check whether a bigger `--imgsz` or `--tiles` got passed.
- `build_field_workbook.py` needs `openpyxl`, which is in `requirements.txt`. If the Comparison sheet has blank predicted columns, put `South_Platte_Predicted_Trash_Dataset.xlsx` in the same folder and rerun.

---

## Rebuilding the base model from public data

Optional, and not something the project needs. Only if you want to make a new `best.pt` from scratch rather than fine tune the one on the release.

`best.pt` was trained on about 43,500 images from public sets, all collapsed to one class:

| Dataset | Where | What it added |
|---|---|---|
| TACO | tacodataset.org, annotations and image URLs on GitHub | about 1,500 ground level litter photos with boxes |
| TrashNet | github.com/garythung/trashnet | 2,500 sorted waste photos, one item per image |
| Waste Classification | kaggle.com/datasets/techsash/waste-classification-data | about 22,500 photos, one item per image |
| UAVVaste | github.com/UAVVaste/UAVVaste | about 770 aerial drone images with boxes, the closest thing to a bridge camera view |

`train_mega.py` downloads and converts these, but it was written for the old 20 class setup and some of its downloads have broken since, so expect to fix paths. The steps that worked:

1. Run `train_mega.py` far enough to download and convert the sets into `mega_training/combined_dataset/`. It needs a Kaggle API token in `~/.kaggle/`.
2. Run `convert_uavvaste_singleclass.py` on the UAVVaste annotations and copy the result in.
3. Run `collapse_to_single_class.py` on the train and val label folders to force every class id to 0.
4. Point `trash_single.yaml` at the folder and train from `yolov8m.pt` for 40 to 50 epochs at 640. On an M4 Max that is about 42 hours.

The classification sets have no boxes, so `train_mega.py` draws one box covering 90 percent of each image. That is a known weakness of the base model and part of why the river fine tune mattered.
