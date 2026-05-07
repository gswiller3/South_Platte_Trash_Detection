# South Platte River Trash Detection System

Computer vision system for detecting and tracking trash in the South Platte River corridor, developed as part of a Colorado School of Mines senior capstone project in partnership with Denver Trout Unlimited.

## Project Overview

This system uses YOLOv8 object detection to automatically identify litter in trail camera images, enabling the creation of pollution heat maps along the 35-mile South Platte corridor from Chatfield Reservoir to Commerce City.

## Features

- Custom-trained YOLOv8 model for trash detection (bottles, plastic bags, wrappers, cans, etc.)
- Rock false positive filtering using color saturation and aspect ratio analysis
- Batch processing for trail camera image folders
- Annotated output images with bounding boxes and confidence scores
- JSON detection logs for data analysis

## Files

| File | Description |
|------|-------------|
| `trash_detector_v3_1.py` | Main detection script with rock filtering |
| `train_mega.py` | Training script for custom model (50,000+ images) |
| `quickstart.py` | Simple script to run detection on image folders |

## Installation

```bash
pip install ultralytics opencv-python numpy tqdm
```

## Usage

### Basic Detection

```bash
python quickstart.py ./input_images ./output_results
```

### Detection with Custom Model

```bash
python trash_detector_v3_1.py ./input_images ./output_results --model ./trained_models/mega_trash_v4/weights/best.pt
```

### Training a New Model

```bash
python train_mega.py
```

## Model Performance

Current model metrics (trained on 50,000+ images):

| Class | mAP50 |
|-------|-------|
| Bottle | 61.3% |
| Cardboard | 82.8% |
| Wrapper | 44.9% |
| Plastic Bag | 22.8% |
| Can | 38.3% |

## Detection Classes

The model detects 20 trash categories:
- plastic_bag, wrapper, bottle, bottle_cap, can
- cup, styrofoam, container, carton, straw
- utensil, paper, cardboard, cigarette, lid
- plastic_film, foil, rope, plastic_other, other_trash

## Data Sources

Training data compiled from:
- TACO (Trash Annotations in Context)
- TrashNet
- Kaggle Waste Classification

## Tools Used

- **Roboflow**: Image annotation and model hosting
- **Kaggle**: Dataset acquisition
- **Ultralytics YOLOv8**: Object detection framework

## License

This project is part of a Colorado School of Mines capstone project.

## Contact

Griffin Willer - Colorado School of Mines
