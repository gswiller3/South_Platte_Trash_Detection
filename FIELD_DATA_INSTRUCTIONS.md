# Field Data Processing Instructions
## South Platte Trash Detection, First Collection

Three scripts. `process_camera.py` runs one SD card end to end. `score_review.py` turns your manual check into precision and recall numbers you can defend. `build_field_workbook.py` turns every camera folder into one Excel workbook.

Everything lands in one top folder called `field_data` with one subfolder per camera. Never mix cameras in one folder.

```
~/Desktop/Trash_Gang/
    best_river.pt                the river model, from the GitHub release
    process_camera.py
    score_review.py
    build_field_workbook.py
    South_Platte_Field_Results.xlsx   built by build_field_workbook.py
    field_data/
        camera_11/
            raw/                 untouched copies off the card
            annotated/           every image that had a detection, boxes drawn
            showcase/            top 10 by confidence, for Blake
            review/
                positives/       50 random annotated images to check
                negatives/       50 random blank images to check
                review_sheet.csv
            detections.csv       one row per box, with camera id and timestamp
            summary.txt          the numbers for this camera
        camera_10/
        ...
```

---

## Part A. Set up (do once)

1. Open Terminal.

2. Go to the project folder.
   ```
   cd ~/Desktop/Trash_Gang
   ```

3. Put the model where the script expects it. Download `best_river.pt` from https://github.com/gswiller3/South_Platte_Trash_Detection/releases/tag/v6-river and drop it in the project folder. It is the river fine tuned model, about 52 MB, and the script looks for it by that name.

   **QUESTION A1:** If the script says `Model file not found: best_river.pt`, run `ls -la ~/Desktop/Trash_Gang/*.pt` and tell me what comes back.

4. Move the scripts into the folder.
   ```
   mv ~/Downloads/process_camera.py .
   mv ~/Downloads/score_review.py .
   mv ~/Downloads/build_field_workbook.py .
   ```

5. Confirm the model is single class.
   ```
   python3 -c "from ultralytics import YOLO; print(YOLO('best_river.pt').names)"
   ```
   You want `{0: 'trash'}`. If you see 20 class names, you grabbed the old v4 model. Stop and find the right one. Do not use plain `best.pt` on river frames, it boxes ripples.

---

## Part B. Process camera 11

6. Plug in the SD card for camera 11.

7. Find its mount name.
   ```
   ls /Volumes
   ```
   You will see something like `GARDEPRO` or `NO NAME` or `Untitled`. Use whatever shows up. If two cards have the same name, only plug in one at a time.

   **QUESTION B1:** Tell me the exact volume name and run `ls "/Volumes/<name>"` then `ls "/Volumes/<name>/DCIM"` and paste what you see. I want to know the folder layout and what the files are named, like `IMAG0001.JPG` or `DSCF0001.JPG`. The script walks every subfolder so it does not matter for running, but I want it on record.

8. Quick test on the first 200 images so you know it works and how fast it runs. Replace `GARDEPRO` with your volume name.
   ```
   python3 process_camera.py /Volumes/GARDEPRO 11 --limit 200
   ```
   Watch the output. You should see:
   - a count of images and videos found
   - `model classes: {0: 'trash'}`
   - a progress line every 100 images
   - a summary block at the end

   **QUESTION B2:** How long did the 200 take? Look at the Terminal timestamp or just time it. Multiply by the total image count divided by 200 to know how long the full card takes. Tell me the number so I can tell you whether to run overnight.

   **QUESTION B3:** Did it report any videos? If so, how many? The cameras may have been set to photo plus video. Videos are skipped for now.

9. Delete the test run so the real run starts clean.
   ```
   rm -rf field_data/camera_11
   ```

10. Run the full card. Two flags matter: `--daylight-only` skips the grayscale IR night frames, which the model was never trained on and which it confuses with moonlight on the water. `--skip-first N` drops the setup frames of you installing the camera. Count how many setup frames there are by looking at the start of the raw folder, then use that number.
    ```
    python3 process_camera.py /Volumes/GARDEPRO 11 --daylight-only --skip-first 60
    ```
    If this is going to take more than an hour, start it and walk away. Do not let the Mac sleep. System Settings, Battery, turn off sleep while plugged in, or run `caffeinate -i python3 process_camera.py /Volumes/GARDEPRO 11` which keeps the Mac awake until the script finishes.

11. When it finishes, read the summary.
    ```
    cat field_data/camera_11/summary.txt
    ```

    **QUESTION B4:** Paste the whole summary block to me. I want images processed, images with trash, total boxes, new trash, and the date range.

12. Eject the card. Do not format it yet. The raw copies are in `field_data/camera_11/raw` but keep the card as a backup until all 11 are done and backed up to a second drive.

---

## Part C. Photos for Blake

13. Open the showcase folder.
    ```
    open field_data/camera_11/showcase
    ```
    These are the 10 highest confidence detections with boxes drawn. Yellow box means new trash. Orange box with `(repeat)` means the same piece was already counted in a previous frame.

14. Look through them. Pick 3 to 5 that are clearly trash and clearly on or near the water. Skip any that look like a false positive even if the confidence is high.

    **QUESTION C1:** Tell me which filenames you picked and whether any of the 10 were obviously wrong. If more than 3 of the 10 are wrong, that is a sign the model needs the fine tune pass on river backgrounds and I want to know before tomorrow.

15. If you want a wider pick, the full annotated set is in `field_data/camera_11/annotated`. Sort by date modified in Finder to see them in capture order.

16. For the meeting, you have three things to show:
    - 3 to 5 showcase images
    - `summary.txt` numbers
    - the review process in Part D, even if you have not finished it yet. Blake should hear that you are validating the counts by hand before you call them final.

---

## Part D. Manual review (this is what makes the numbers defensible)

The script already pulled two random samples with a fixed seed so anyone can reproduce them:
- 50 images the model flagged as trash, saved with boxes in `review/positives`
- 50 images the model said were empty, saved raw in `review/negatives`

Checking the positives tells you precision. Checking the negatives tells you what the model is missing. Both matter. Because most images are negatives, even a small miss rate on negatives can drag recall down a lot. That is the number you most need to know before you trust per camera counts.

17. Open the review sheet in Excel or Numbers.
    ```
    open field_data/camera_11/review/review_sheet.csv
    ```
    Columns: camera_id, filename, stratum, model_boxes, human_trash_count, verdict, notes.

18. Open the positives folder next to it.
    ```
    open field_data/camera_11/review/positives
    ```
    For each image, look at what is inside the boxes. Fill in:
    - `human_trash_count`: how many real pieces of trash you see in the whole image
    - `verdict`: `correct` if the boxes are on real trash, `false_pos` if the model boxed a leaf, foam, shadow, rock, bird, or anything that is not trash
    - `notes`: optional, what it actually was if it was wrong

19. Open the negatives folder.
    ```
    open field_data/camera_11/review/negatives
    ```
    These have no boxes. Look hard, the whole frame, especially edges and small items. Fill in:
    - `human_trash_count`: how many pieces of trash you see
    - `verdict`: `correct` if there is truly no trash, `false_neg` if you see trash the model missed
    - `notes`: what it was and roughly where

20. Save the sheet as CSV, same name, same place. Excel will nag about losing formatting. That is fine.

21. Score it.
    ```
    python3 score_review.py 11
    ```
    You get image level precision, the miss rate on negatives, and an estimated recall, each with a 95 percent confidence interval. It also writes `review/validation_result.txt`.

    **QUESTION D1:** Paste the whole output to me. If precision is under 80 percent or the miss rate is over 10 percent, we talk about raising the confidence threshold or doing the fine tune before the other 10 cards.

22. If you want tighter error bars, rerun step 10 with `--review-n 100`. That doubles the review work but roughly cuts the confidence interval by 30 percent. Fifty is fine for tomorrow.

---

## Part E. Remaining cameras

23. Same as Part B for each card. Change only the camera number.
    ```
    python3 process_camera.py /Volumes/GARDEPRO 10
    python3 process_camera.py /Volumes/GARDEPRO 9
    ```
    and so on. Each card gets its own subfolder. Nothing from one camera touches another.

24. You do not have to do the manual review on every card. Do it on camera 11 now. Then do it on two or three more that look different from each other, like one upstream, one downtown, one with a lot of detections. If the precision and miss rate hold steady across those, apply the camera 11 numbers to the rest and say so in the report.

25. After every card, rebuild the workbook. It reads every `detections.csv`, `summary.txt` and scored review in `field_data/` and writes `South_Platte_Field_Results.xlsx` plus two CSV files for the heat map tool. Nothing is typed in by hand.
    ```
    python3 build_field_workbook.py
    ```
    Put `South_Platte_Predicted_Trash_Dataset.xlsx` in the same folder first if you want the Comparison sheet to show predicted next to observed. The camera clocks say 2025, the script fixes the year to 2026 on its own and says so on the Cameras sheet.

    **QUESTION E4:** Send me the workbook once three or more cameras are in it.

---

## How the counting works

**Per camera attribution.** You type the camera number on the command line. It is written into every row of `detections.csv` and the folder name. There is no way for camera 11 detections to end up attributed to camera 10 unless you type the wrong number. Double check the number against the label on the SD card before you hit enter.

**QUESTION E1:** The budget says 12 camera stations. You said 11 cards. Was one camera not deployed, was a card lost, or is one still out there? I need to know so the per camera table has the right rows.

**Not counting the same trash twice.** Updated Oct 20. The script sorts images by timestamp and tags every box one of three ways:
- `stationary`: overlaps a box from any of the last 30 frames (5 minutes at 10 s) by more than 50 percent. Trash snagged on a rock or a pier, counted once, then ignored while it sits there.
- `drift`: its center is within 35 percent of the frame diagonal of an unmatched box in the previous frame, and that frame is no more than 25 s old. The same item floating through, counted once on the frame it first appeared.
- `new`: neither. This is the count.

`summary.txt` reports all four lines: `Total trash boxes`, `Stationary repeats`, `Drift repeats`, `New trash (repeats removed)`. Show Blake the last one. The first three exist so the count is not hiding anything. `detections.csv` has a `repeat_type` column per box.

Known gaps: an item that flickers out of detection for more than 5 minutes and comes back is counted again. Two different items in the same frame close together can be merged as one drift. An item that circulates in an eddy and leaves the frame for a while is counted again when it returns. None of these are big at low flow with sparse trash. Reprocess any card you already ran so every camera uses the same rule. The old 5 frame version overcounted drifting items two to three times each.

**QUESTION E2:** Were the cameras set to time lapse or motion trigger, and what interval? If time lapse, how many seconds between shots? This decides whether 5 frames of lookback is right or whether it needs to be longer.

**Night frames.** The model was trained on daylight only. On the first card it boxed moonlight reflections as trash. The `--daylight-only` flag detects IR frames by checking whether the color channels are nearly identical, which is what a grayscale IR shot looks like. It does not use the clock, since the camera clock was off by a year. The summary reports how many night frames were skipped so the exclusion is on record. For Blake: night detection is scoped out of phase one and would need IR training data to do properly. Trash that enters at night is still there at sunrise unless it floats all the way through in the dark, and that is a stated limitation.

**Setup frames.** The first frames on each card are you installing the camera. `--skip-first N` drops them so a picture of your hand does not count as trash. The summary reports N.

---

## If something breaks

- `ModuleNotFoundError: ultralytics` means run `pip3 install ultralytics opencv-python pillow`
- `No images found` means the volume name is wrong or the card is empty. Check `ls /Volumes` again.
- `could not read IMAG0045.JPG` on a handful of files is a corrupt image. Normal for trail cams. The script skips it and moves on. If it is more than a few percent of the card, tell me.
- If the Mac fans go full speed and the progress stops, the MPS backend choked on the image size. The default is 640 on the whole frame, which is light, so check whether a bigger `--imgsz` or `--tiles` got passed. Detection size has to stay at 640 because that is what the model was trained at.
- `Model file not found: best_river.pt` means the model is not in the folder you are running from. See step 3.
- If you need to rerun without recopying from the card, point the script at the raw folder:
  ```
  python3 process_camera.py field_data/camera_11/raw 11 --skip-copy
  ```

---

## Questions to answer back, all in one place

- A1: what .pt files are in the folder, if the script cannot find best_river.pt
- B1: volume name, folder layout, file naming on the card
- B2: how long 200 images took
- B3: any videos on the card
- B4: the full summary block for camera 11
- C1: which showcase images you picked, how many of the 10 were wrong
- D1: the full score_review output
- E1: 11 cards vs 12 stations
- E2: time lapse or motion trigger, and the interval
- E3: any night or IR images
- E4: the workbook once three or more cameras are in it
