#!/usr/bin/env python3
"""
MEGA TRASH DETECTION TRAINER v4.0
=================================
Downloads and combines multiple public datasets:
- TACO (existing) - 1,456 images
- TrashNet - 2,527 images  
- Drinking Waste - 9,640 images
- UAVVaste - 3,716 images
- Trash-ICRA19 - 5,700 images (underwater!)

Total: ~23,000 images

Training time: 10-15 hours on M4 Max

Author: South Platte Capstone Team
"""

import os
import sys
import json
import shutil
import zipfile
import tarfile
import requests
import subprocess
from pathlib import Path
from collections import defaultdict
import random
import glob

def check_deps():
    required = ['ultralytics', 'cv2', 'PIL', 'numpy', 'tqdm', 'requests', 'kaggle']
    missing = []
    for mod in required:
        try:
            if mod == 'cv2':
                __import__('cv2')
            elif mod == 'PIL':
                __import__('PIL')
            else:
                __import__(mod)
        except ImportError:
            pkg = 'opencv-python' if mod == 'cv2' else 'Pillow' if mod == 'PIL' else mod
            missing.append(pkg)
    if missing:
        print(f"Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing + ['-q'])
        print("Restart script.")
        sys.exit(0)

check_deps()

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import kaggle


# ============================================================================
# CONFIGURATION
# ============================================================================

class MegaConfig:
    # Paths
    BASE_DIR = Path("./mega_training")
    DOWNLOAD_DIR = BASE_DIR / "downloads"
    COMBINED_DIR = BASE_DIR / "combined_dataset"
    MODEL_DIR = Path("./trained_models")
    EXISTING_TACO = Path("./taco_training/dataset")
    
    # Training settings
    MODEL_BASE = "yolov8m.pt"
    EPOCHS = 150              # More epochs for larger dataset
    IMAGE_SIZE = 640
    BATCH_SIZE = 8
    PATIENCE = 25             # More patience
    
    # Unified classes for all datasets
    UNIFIED_CLASSES = [
        'plastic_bag',      # 0
        'wrapper',          # 1
        'bottle',           # 2
        'bottle_cap',       # 3
        'can',              # 4
        'cup',              # 5
        'styrofoam',        # 6
        'container',        # 7
        'carton',           # 8
        'straw',            # 9
        'utensil',          # 10
        'paper',            # 11
        'cardboard',        # 12
        'cigarette',        # 13
        'lid',              # 14
        'plastic_film',     # 15
        'foil',             # 16
        'rope',             # 17
        'plastic_other',    # 18
        'other_trash',      # 19
    ]
    
    # Class mappings for each dataset
    TRASHNET_MAPPING = {
        'cardboard': 'cardboard',
        'glass': 'bottle',
        'metal': 'can',
        'paper': 'paper',
        'plastic': 'plastic_other',
        'trash': 'other_trash',
    }
    
    DRINKING_WASTE_MAPPING = {
        'aluminium_can': 'can',
        'aluminum_can': 'can',
        'can': 'can',
        'glass_bottle': 'bottle',
        'glass': 'bottle',
        'pet_bottle': 'bottle',
        'pet': 'bottle',
        'plastic_bottle': 'bottle',
        'bottle': 'bottle',
        'hdpe_bottle': 'bottle',
        'hdpe': 'bottle',
        'tetra_pack': 'carton',
        'tetra': 'carton',
        'carton': 'carton',
    }


# ============================================================================
# DATASET DOWNLOADERS
# ============================================================================

class DatasetDownloader:
    def __init__(self, config: MegaConfig):
        self.config = config
        self.config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    def download_trashnet(self) -> Path:
        """Download TrashNet from GitHub."""
        print("\n" + "="*60)
        print("📥 DOWNLOADING TRASHNET (2,527 images)")
        print("="*60)
        
        trashnet_dir = self.config.DOWNLOAD_DIR / "trashnet"
        
        if (trashnet_dir / "dataset-resized").exists():
            print("   Already downloaded.")
            return trashnet_dir / "dataset-resized"
        
        url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
        zip_path = self.config.DOWNLOAD_DIR / "trashnet.zip"
        
        print("   Downloading from GitHub...")
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            total = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f:
                with tqdm(total=total, unit='B', unit_scale=True, desc="   TrashNet") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print("   Extracting...")
            trashnet_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(trashnet_dir)
            
            print("   ✓ TrashNet downloaded")
            return trashnet_dir / "dataset-resized"
            
        except Exception as e:
            print(f"   ⚠️ TrashNet download failed: {e}")
            return None
    
    def download_drinking_waste(self) -> Path:
        """Download Drinking Waste Classification from Kaggle."""
        print("\n" + "="*60)
        print("📥 DOWNLOADING DRINKING WASTE (9,640 images)")
        print("="*60)
        
        drinking_dir = self.config.DOWNLOAD_DIR / "drinking_waste"
        
        if drinking_dir.exists() and any(drinking_dir.glob("**/*.jpg")):
            print("   Already downloaded.")
            return drinking_dir
        
        drinking_dir.mkdir(exist_ok=True)
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            print("   Downloading from Kaggle...")
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                'arkadiyhacks/drinking-waste-classification',
                path=str(drinking_dir),
                unzip=True
            )
            print("   ✓ Drinking Waste downloaded")
            return drinking_dir
        except Exception as e:
            print(f"   ⚠️ Download failed: {e}")
            return None
    
    def download_uavvaste(self) -> Path:
        """Download UAVVaste dataset."""
        print("\n" + "="*60)
        print("📥 DOWNLOADING UAVVASTE (3,716 aerial images)")
        print("="*60)
        
        uav_dir = self.config.DOWNLOAD_DIR / "uavvaste"
        
        if uav_dir.exists() and any(uav_dir.glob("**/*.jpg")):
            print("   Already downloaded.")
            return uav_dir
        
        uav_dir.mkdir(exist_ok=True)
        
        # UAVVaste is typically available through academic channels
        # Try GitHub mirror
        url = "https://github.com/UAVVaste/UAVVaste/archive/refs/heads/main.zip"
        
        try:
            print("   Downloading from GitHub...")
            response = requests.get(url, stream=True, timeout=120)
            
            if response.status_code == 404:
                print("   ⚠️ UAVVaste not available on GitHub")
                print("   Skipping UAVVaste...")
                return None
            
            response.raise_for_status()
            
            zip_path = self.config.DOWNLOAD_DIR / "uavvaste.zip"
            total = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                with tqdm(total=total, unit='B', unit_scale=True, desc="   UAVVaste") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print("   Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(uav_dir)
            
            print("   ✓ UAVVaste downloaded")
            return uav_dir
            
        except Exception as e:
            print(f"   ⚠️ UAVVaste download failed: {e}")
            return None
    
    def download_trash_icra(self) -> Path:
        """Download Trash-ICRA19 (underwater trash)."""
        print("\n" + "="*60)
        print("📥 DOWNLOADING TRASH-ICRA19 (underwater trash)")
        print("="*60)
        
        icra_dir = self.config.DOWNLOAD_DIR / "trash_icra"
        
        if icra_dir.exists() and any(icra_dir.glob("**/*.jpg")):
            print("   Already downloaded.")
            return icra_dir
        
        icra_dir.mkdir(exist_ok=True)
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            print("   Downloading from Kaggle...")
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                'jkruppa/trash-icra19',
                path=str(icra_dir),
                unzip=True
            )
            print("   ✓ Trash-ICRA19 downloaded")
            return icra_dir
        except Exception as e:
            print(f"   ⚠️ Trash-ICRA19 download failed: {e}")
            return None
    
    def download_waste_pictures(self) -> Path:
        """Download Waste Classification data from Kaggle."""
        print("\n" + "="*60)
        print("📥 DOWNLOADING WASTE CLASSIFICATION (additional images)")
        print("="*60)
        
        waste_dir = self.config.DOWNLOAD_DIR / "waste_classification"
        
        if waste_dir.exists() and any(waste_dir.glob("**/*.jpg")):
            print("   Already downloaded.")
            return waste_dir
        
        waste_dir.mkdir(exist_ok=True)
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            print("   Downloading from Kaggle...")
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                'techsash/waste-classification-data',
                path=str(waste_dir),
                unzip=True
            )
            print("   ✓ Waste Classification downloaded")
            return waste_dir
        except Exception as e:
            print(f"   ⚠️ Waste Classification download failed: {e}")
            return None


# ============================================================================
# DATASET CONVERTER (to YOLO format)
# ============================================================================

class DatasetConverter:
    def __init__(self, config: MegaConfig):
        self.config = config
        self.image_counter = 0
    
    def convert_classification_to_detection(self, 
                                            dataset_dir: Path, 
                                            class_mapping: dict,
                                            output_images: Path,
                                            output_labels: Path,
                                            dataset_name: str) -> int:
        """
        Convert classification dataset (folder per class) to YOLO detection format.
        Creates pseudo-bounding boxes (full image = single object).
        """
        if dataset_dir is None:
            return 0
        
        print(f"\n   Converting {dataset_name}...")
        
        converted = 0
        
        # Find all class folders
        for class_folder in dataset_dir.iterdir():
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name.lower().replace(' ', '_')
            
            # Map to unified class
            unified_class = class_mapping.get(class_name)
            if unified_class is None:
                # Try partial match
                for key, value in class_mapping.items():
                    if key in class_name or class_name in key:
                        unified_class = value
                        break
            
            if unified_class is None:
                unified_class = 'other_trash'
            
            try:
                class_idx = self.config.UNIFIED_CLASSES.index(unified_class)
            except ValueError:
                class_idx = 19  # other_trash
            
            # Process images in this class folder
            for img_path in class_folder.glob("*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                
                try:
                    # Verify image is valid
                    img = Image.open(img_path)
                    img.verify()
                    
                    # Copy image with unique name
                    new_name = f"{dataset_name}_{self.image_counter:06d}"
                    new_img_path = output_images / f"{new_name}.jpg"
                    
                    # Convert to JPG if needed
                    img = Image.open(img_path).convert('RGB')
                    img.save(new_img_path, 'JPEG', quality=95)
                    
                    # Create YOLO label (full image bounding box)
                    # For classification datasets, we assume the object fills most of the image
                    # YOLO format: class_id x_center y_center width height (normalized)
                    label_content = f"{class_idx} 0.5 0.5 0.9 0.9\n"
                    
                    label_path = output_labels / f"{new_name}.txt"
                    with open(label_path, 'w') as f:
                        f.write(label_content)
                    
                    self.image_counter += 1
                    converted += 1
                    
                except Exception as e:
                    continue
        
        print(f"      ✓ Converted {converted} images from {dataset_name}")
        return converted
    
    def copy_existing_yolo_dataset(self, 
                                   dataset_dir: Path,
                                   output_images: Path,
                                   output_labels: Path,
                                   dataset_name: str) -> int:
        """Copy existing YOLO format dataset."""
        if dataset_dir is None or not dataset_dir.exists():
            return 0
        
        print(f"\n   Copying {dataset_name}...")
        
        copied = 0
        
        # Find images and labels dirs
        img_dirs = list(dataset_dir.glob("**/images"))
        lbl_dirs = list(dataset_dir.glob("**/labels"))
        
        for img_dir in img_dirs:
            # Find corresponding label dir
            lbl_dir = img_dir.parent / "labels"
            if not lbl_dir.exists():
                continue
            
            for img_path in img_dir.glob("*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                
                if not lbl_path.exists():
                    continue
                
                try:
                    # Copy with unique name
                    new_name = f"{dataset_name}_{self.image_counter:06d}"
                    
                    shutil.copy2(img_path, output_images / f"{new_name}{img_path.suffix}")
                    shutil.copy2(lbl_path, output_labels / f"{new_name}.txt")
                    
                    self.image_counter += 1
                    copied += 1
                    
                except Exception as e:
                    continue
        
        print(f"      ✓ Copied {copied} images from {dataset_name}")
        return copied


# ============================================================================
# DATASET COMBINER
# ============================================================================

class MegaDatasetCombiner:
    def __init__(self, config: MegaConfig):
        self.config = config
        self.downloader = DatasetDownloader(config)
        self.converter = DatasetConverter(config)
    
    def build_mega_dataset(self) -> Path:
        """Download, convert, and combine all datasets."""
        
        print("\n" + "="*60)
        print("🔨 BUILDING MEGA DATASET")
        print("="*60)
        
        # Create output directories
        train_images = self.config.COMBINED_DIR / "train" / "images"
        train_labels = self.config.COMBINED_DIR / "train" / "labels"
        val_images = self.config.COMBINED_DIR / "val" / "images"
        val_labels = self.config.COMBINED_DIR / "val" / "labels"
        
        for d in [train_images, train_labels, val_images, val_labels]:
            d.mkdir(parents=True, exist_ok=True)
        
        all_images = []
        
        # 1. Use existing TACO
        print("\n📂 Using existing TACO dataset...")
        if self.config.EXISTING_TACO.exists():
            taco_count = self.converter.copy_existing_yolo_dataset(
                self.config.EXISTING_TACO,
                train_images, train_labels,
                "taco"
            )
        else:
            print("   ⚠️ TACO not found - run previous training first")
            taco_count = 0
        
        # 2. Download and convert TrashNet
        trashnet_dir = self.downloader.download_trashnet()
        if trashnet_dir:
            self.converter.convert_classification_to_detection(
                trashnet_dir,
                self.config.TRASHNET_MAPPING,
                train_images, train_labels,
                "trashnet"
            )
        
        # 3. Download and convert Drinking Waste
        drinking_dir = self.downloader.download_drinking_waste()
        if drinking_dir:
            # Find the actual data folder
            data_folders = list(drinking_dir.glob("**/DATASET")) or \
                          list(drinking_dir.glob("**/dataset")) or \
                          [drinking_dir]
            
            for data_folder in data_folders:
                self.converter.convert_classification_to_detection(
                    data_folder,
                    self.config.DRINKING_WASTE_MAPPING,
                    train_images, train_labels,
                    "drinking"
                )
        
        # 4. Download Waste Classification
        waste_dir = self.downloader.download_waste_pictures()
        if waste_dir:
            # This dataset has TRAIN and TEST folders
            for split_folder in waste_dir.glob("**/TRAIN"):
                self.converter.convert_classification_to_detection(
                    split_folder,
                    self.config.TRASHNET_MAPPING,  # Similar classes
                    train_images, train_labels,
                    "waste"
                )
        
        # 5. Try Trash-ICRA19 (underwater)
        icra_dir = self.downloader.download_trash_icra()
        if icra_dir:
            self.converter.copy_existing_yolo_dataset(
                icra_dir,
                train_images, train_labels,
                "icra"
            )
        
        # 6. Try UAVVaste
        uav_dir = self.downloader.download_uavvaste()
        if uav_dir:
            self.converter.copy_existing_yolo_dataset(
                uav_dir,
                train_images, train_labels,
                "uav"
            )
        
        # Create train/val split
        print("\n📂 Creating train/val split...")
        self._create_split(train_images, train_labels, val_images, val_labels)
        
        # Create YAML config
        yaml_path = self._create_yaml()
        
        # Count final dataset
        train_count = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
        val_count = len(list(val_images.glob("*.jpg"))) + len(list(val_images.glob("*.png")))
        
        print(f"\n✓ MEGA DATASET COMPLETE")
        print(f"   Training images: {train_count}")
        print(f"   Validation images: {val_count}")
        print(f"   Total: {train_count + val_count}")
        
        return yaml_path
    
    def _create_split(self, train_img: Path, train_lbl: Path, val_img: Path, val_lbl: Path, val_ratio: float = 0.15):
        """Move some training images to validation."""
        all_images = list(train_img.glob("*.*"))
        random.shuffle(all_images)
        
        val_count = int(len(all_images) * val_ratio)
        val_images = all_images[:val_count]
        
        for img_path in tqdm(val_images, desc="   Creating val split"):
            lbl_path = train_lbl / f"{img_path.stem}.txt"
            
            # Move to val
            shutil.move(str(img_path), str(val_img / img_path.name))
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(val_lbl / lbl_path.name))
    
    def _create_yaml(self) -> Path:
        """Create YOLO dataset config."""
        yaml_content = f"""# MEGA Trash Detection Dataset
# Combined from: TACO + TrashNet + Drinking Waste + More
# Auto-generated by train_mega.py

path: {self.config.COMBINED_DIR.absolute()}
train: train/images
val: val/images

nc: {len(self.config.UNIFIED_CLASSES)}

names:
"""
        for i, name in enumerate(self.config.UNIFIED_CLASSES):
            yaml_content += f"  {i}: {name}\n"
        
        yaml_path = self.config.BASE_DIR / "mega_dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        return yaml_path


# ============================================================================
# TRAINER
# ============================================================================

def train_mega_model(yaml_path: Path, config: MegaConfig) -> Path:
    """Train on the mega combined dataset."""
    
    print("\n" + "="*60)
    print("🚀 STARTING MEGA TRAINING")
    print("="*60)
    print(f"\nModel: {config.MODEL_BASE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Image size: {config.IMAGE_SIZE}")
    print("\n⏰ Estimated time: 10-15 hours")
    print("   Go to sleep, this will run overnight!\n")
    
    model = YOLO(config.MODEL_BASE)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    results = model.train(
        data=str(yaml_path),
        epochs=config.EPOCHS,
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        patience=config.PATIENCE,
        project=str(config.MODEL_DIR),
        name="mega_trash_v4",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,
        cos_lr=True,
        amp=True,
        close_mosaic=15,
        mosaic=1.0,
        mixup=0.1,           # Add mixup augmentation
        copy_paste=0.1,      # Add copy-paste augmentation
        degrees=10,          # Slight rotation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    
    # Copy best model
    best_src = config.MODEL_DIR / "mega_trash_v4" / "weights" / "best.pt"
    best_dst = config.MODEL_DIR / "mega_trash_v4_best.pt"
    
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n✅ Training complete!")
        print(f"   Best model: {best_dst}")
    
    return best_dst


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("  🗑️  MEGA TRASH DETECTION TRAINER v4.0  🗑️")
    print("  ~23,000 images from multiple datasets")
    print("="*60)
    
    config = MegaConfig()
    
    # Check for Kaggle credentials
    kaggle_token = Path.home() / ".kaggle" / "access_token"
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_token.exists() and not kaggle_json.exists():
        print("\n⚠️  Kaggle credentials not found!")
        print("   Run: mkdir -p ~/.kaggle && echo YOUR_TOKEN > ~/.kaggle/access_token")
        print("   Then restart this script.")
        sys.exit(1)
    
    print("\n✓ Kaggle credentials found")
    
    # Build mega dataset
    combiner = MegaDatasetCombiner(config)
    yaml_path = combiner.build_mega_dataset()
    
    # Train
    model_path = train_mega_model(yaml_path, config)
    
    print("\n" + "="*60)
    print("  🎉 MEGA TRAINING COMPLETE! 🎉")
    print("="*60)
    print(f"\n📦 Your new model: {model_path}")
    print("\nTo test:")
    print(f"  python3 trash_detector_v3_1.py ./test_images ./results --model {model_path}")


if __name__ == '__main__':
    main()
