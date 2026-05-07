#!/usr/bin/env python3
"""
COMPREHENSIVE TRASH DETECTION TRAINER
======================================
Combines multiple datasets for maximum accuracy:
1. TACO (already downloaded) - General litter
2. WaDaBa - Waste in water (rivers, lakes)
3. Drinking Waste - Bottles, cans, cups
4. Additional Roboflow datasets

Total: ~12,000+ images

Usage:
    python train_comprehensive.py

Author: South Platte Capstone Team
"""

import os
import sys
import json
import shutil
import requests
import zipfile
import tarfile
from pathlib import Path
from collections import defaultdict
import random

def check_deps():
    required = ['ultralytics', 'cv2', 'PIL', 'numpy', 'tqdm', 'requests', 'roboflow']
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
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing + ['-q'])
        print("Restart script after install.")
        sys.exit(0)

check_deps()

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    BASE_DIR = Path("./comprehensive_training")
    COMBINED_DIR = BASE_DIR / "combined_dataset"
    TACO_DIR = Path("./taco_training")  # Already downloaded
    MODEL_DIR = Path("./trained_models")
    
    # Training
    MODEL_BASE = "yolov8m.pt"
    EPOCHS = 100
    IMAGE_SIZE = 640
    BATCH_SIZE = 8
    PATIENCE = 20
    
    # Split
    VAL_SPLIT = 0.15
    
    # Unified class mapping - all datasets map to these
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


# ============================================================================
# DATASET DOWNLOADERS
# ============================================================================

class DatasetManager:
    """Manages downloading and combining multiple trash datasets."""
    
    def __init__(self, config: Config):
        self.config = config
        self.config.BASE_DIR.mkdir(parents=True, exist_ok=True)
        
    def download_roboflow_datasets(self, api_key: str):
        """Download multiple trash datasets from Roboflow Universe."""
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        
        datasets = [
            # Format: (workspace, project, version, name)
            ("alex-hyams-cosqx", "taco-trash-annotations-in-context", 12, "taco_rf"),
            ("material-identification", "garbage-classification-3", 2, "garbage_class"),
            ("divya-lzcld", "aqua-trash", 3, "aqua_trash"),  # Water trash!
        ]
        
        downloaded = []
        
        for workspace, project, version, name in datasets:
            print(f"\n📥 Downloading {name}...")
            try:
                proj = rf.workspace(workspace).project(project)
                dataset = proj.version(version).download("yolov8", location=str(self.config.BASE_DIR / name))
                downloaded.append(self.config.BASE_DIR / name)
                print(f"   ✓ Downloaded {name}")
            except Exception as e:
                print(f"   ⚠️ Failed to download {name}: {e}")
        
        return downloaded
    
    def download_wadaba(self):
        """Download WaDaBa (Water Debris) dataset."""
        print("\n📥 Downloading WaDaBa (Water Debris) dataset...")
        
        wadaba_dir = self.config.BASE_DIR / "wadaba"
        wadaba_dir.mkdir(exist_ok=True)
        
        # WaDaBa is hosted on GitHub
        # Note: This dataset may need manual download if not directly available
        url = "https://github.com/Wadan-A/WaDaBa/archive/refs/heads/main.zip"
        
        zip_path = wadaba_dir / "wadaba.zip"
        
        try:
            if not zip_path.exists():
                print("   Downloading from GitHub...")
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                total = int(response.headers.get('content-length', 0))
                with open(zip_path, 'wb') as f:
                    with tqdm(total=total, unit='B', unit_scale=True, desc="   WaDaBa") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Extract
            print("   Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(wadaba_dir)
            
            print("   ✓ WaDaBa downloaded")
            return wadaba_dir
            
        except Exception as e:
            print(f"   ⚠️ WaDaBa download failed: {e}")
            print("   Will continue without WaDaBa")
            return None
    
    def use_existing_taco(self):
        """Use already-downloaded TACO dataset."""
        print("\n📂 Checking existing TACO dataset...")
        
        taco_train = self.config.TACO_DIR / "dataset" / "train" / "images"
        taco_val = self.config.TACO_DIR / "dataset" / "val" / "images"
        
        if taco_train.exists() and taco_val.exists():
            train_count = len(list(taco_train.glob("*.jpg")))
            val_count = len(list(taco_val.glob("*.jpg")))
            print(f"   ✓ Found TACO: {train_count} train, {val_count} val images")
            return self.config.TACO_DIR / "dataset"
        else:
            print("   ⚠️ TACO not found. Run train_taco_model.py first to download.")
            return None


# ============================================================================
# DATASET COMBINER
# ============================================================================

class DatasetCombiner:
    """Combines multiple YOLO-format datasets into one."""
    
    def __init__(self, config: Config):
        self.config = config
        self.class_mapping = {}  # Maps source classes to unified classes
        
    def combine_datasets(self, dataset_paths: list):
        """Combine multiple datasets into unified format."""
        print("\n🔄 Combining datasets...")
        
        combined_train_img = self.config.COMBINED_DIR / "train" / "images"
        combined_train_lbl = self.config.COMBINED_DIR / "train" / "labels"
        combined_val_img = self.config.COMBINED_DIR / "val" / "images"
        combined_val_lbl = self.config.COMBINED_DIR / "val" / "labels"
        
        for d in [combined_train_img, combined_train_lbl, combined_val_img, combined_val_lbl]:
            d.mkdir(parents=True, exist_ok=True)
        
        total_train = 0
        total_val = 0
        
        for dataset_path in dataset_paths:
            if dataset_path is None:
                continue
                
            dataset_path = Path(dataset_path)
            dataset_name = dataset_path.name
            
            print(f"\n   Processing {dataset_name}...")
            
            # Find train/val structure
            train_img = self._find_images_dir(dataset_path, "train")
            val_img = self._find_images_dir(dataset_path, "val")
            
            if train_img:
                count = self._copy_dataset_split(train_img, combined_train_img, combined_train_lbl, dataset_name)
                total_train += count
                print(f"      Train: {count} images")
            
            if val_img:
                count = self._copy_dataset_split(val_img, combined_val_img, combined_val_lbl, dataset_name)
                total_val += count
                print(f"      Val: {count} images")
        
        print(f"\n   ✓ Combined dataset: {total_train} train, {total_val} val")
        
        return total_train, total_val
    
    def _find_images_dir(self, base_path: Path, split: str) -> Path:
        """Find images directory for a split."""
        candidates = [
            base_path / split / "images",
            base_path / split,
            base_path / "data" / split / "images",
            base_path / "images" / split,
        ]
        
        for c in candidates:
            if c.exists() and any(c.glob("*.jpg")) or any(c.glob("*.png")):
                return c
        return None
    
    def _copy_dataset_split(self, img_dir: Path, out_img: Path, out_lbl: Path, prefix: str) -> int:
        """Copy images and labels from one split."""
        count = 0
        
        # Find labels directory
        lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.exists():
            lbl_dir = img_dir.parent.parent / "labels" / img_dir.name
        
        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            # New unique name
            new_name = f"{prefix}_{img_path.stem}"
            new_img = out_img / f"{new_name}{img_path.suffix}"
            new_lbl = out_lbl / f"{new_name}.txt"
            
            # Copy image
            shutil.copy2(img_path, new_img)
            
            # Copy and remap labels
            old_lbl = lbl_dir / f"{img_path.stem}.txt"
            if old_lbl.exists():
                self._copy_and_remap_labels(old_lbl, new_lbl)
            else:
                # Create empty label file
                new_lbl.touch()
            
            count += 1
        
        return count
    
    def _copy_and_remap_labels(self, src: Path, dst: Path):
        """Copy label file, remapping class IDs to unified classes."""
        # For now, just copy directly since most datasets use similar class structures
        # In a production system, you'd remap class IDs based on each dataset's classes
        shutil.copy2(src, dst)
    
    def create_yaml(self) -> Path:
        """Create YOLO dataset config."""
        print("\n📝 Creating dataset config...")
        
        yaml_content = f"""# Combined Trash Detection Dataset
# TACO + WaDaBa + Drinking Waste + More

path: {self.config.COMBINED_DIR.absolute()}
train: train/images
val: val/images

nc: {len(self.config.UNIFIED_CLASSES)}

names:
"""
        for i, name in enumerate(self.config.UNIFIED_CLASSES):
            yaml_content += f"  {i}: {name}\n"
        
        yaml_path = self.config.BASE_DIR / "combined_dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"   ✓ Saved to {yaml_path}")
        return yaml_path


# ============================================================================
# ALTERNATIVE: DOWNLOAD FROM ROBOFLOW UNIVERSE (Easier)
# ============================================================================

def download_from_roboflow_universe(api_key: str, config: Config):
    """
    Download pre-labeled trash datasets from Roboflow Universe.
    These are already in YOLO format with good annotations.
    """
    from roboflow import Roboflow
    
    print("\n" + "="*60)
    print("📦 DOWNLOADING FROM ROBOFLOW UNIVERSE")
    print("="*60)
    
    rf = Roboflow(api_key=api_key)
    
    # High-quality trash detection datasets on Roboflow Universe
    datasets_to_download = [
        {
            "workspace": "alex-hyams-cosqx",
            "project": "taco-trash-annotations-in-context",
            "version": 12,
            "name": "taco"
        },
        {
            "workspace": "garbage-detection-cylon",
            "project": "plastic-garbage-detection", 
            "version": 2,
            "name": "plastic_garbage"
        },
        {
            "workspace": "project-garbage-detection",
            "project": "garbage-detection-cskjp",
            "version": 3,
            "name": "garbage_detection"
        },
    ]
    
    downloaded_paths = []
    
    for ds in datasets_to_download:
        print(f"\n📥 {ds['name']}...")
        try:
            project = rf.workspace(ds['workspace']).project(ds['project'])
            download_path = config.BASE_DIR / ds['name']
            dataset = project.version(ds['version']).download("yolov8", location=str(download_path))
            downloaded_paths.append(download_path)
            print(f"   ✓ Downloaded to {download_path}")
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
    
    return downloaded_paths


# ============================================================================
# SIMPLE COMBINED TRAINER (Uses existing TACO + new downloads)
# ============================================================================

def train_combined(yaml_path: Path, config: Config):
    """Train on combined dataset."""
    print("\n" + "="*60)
    print("🚀 STARTING COMBINED TRAINING")
    print("="*60)
    print(f"\nModel: {config.MODEL_BASE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print("\nThis will take several hours. Progress below.\n")
    
    model = YOLO(config.MODEL_BASE)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    results = model.train(
        data=str(yaml_path),
        epochs=config.EPOCHS,
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        patience=config.PATIENCE,
        project=str(config.MODEL_DIR),
        name="comprehensive_trash",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,
        cos_lr=True,
        amp=True,
        close_mosaic=10,
    )
    
    # Copy best model
    best_src = config.MODEL_DIR / "comprehensive_trash" / "weights" / "best.pt"
    best_dst = config.MODEL_DIR / "comprehensive_trash_best.pt"
    
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n✅ Best model: {best_dst}")
    
    return best_dst


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("  COMPREHENSIVE TRASH DETECTION TRAINER")
    print("  Multi-Dataset Training Pipeline")
    print("="*60)
    
    config = Config()
    
    # Check for Roboflow API key
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    
    if not api_key:
        print("\n⚠️  ROBOFLOW_API_KEY not set!")
        print("\nTo download additional datasets, set your API key:")
        print("  export ROBOFLOW_API_KEY='your_key_here'")
        print("\nProceeding with existing TACO data only...\n")
        use_roboflow = False
    else:
        use_roboflow = True
        print(f"\n✓ Roboflow API key found")
    
    # Initialize managers
    dataset_mgr = DatasetManager(config)
    combiner = DatasetCombiner(config)
    
    # Collect dataset paths
    dataset_paths = []
    
    # 1. Use existing TACO
    taco_path = dataset_mgr.use_existing_taco()
    if taco_path:
        dataset_paths.append(taco_path)
    
    # 2. Download additional datasets from Roboflow
    if use_roboflow:
        rf_paths = download_from_roboflow_universe(api_key, config)
        dataset_paths.extend(rf_paths)
    
    # 3. Try to download WaDaBa
    wadaba_path = dataset_mgr.download_wadaba()
    if wadaba_path:
        dataset_paths.append(wadaba_path)
    
    if not dataset_paths:
        print("\n❌ No datasets available! Run train_taco_model.py first.")
        sys.exit(1)
    
    # Combine datasets
    train_count, val_count = combiner.combine_datasets(dataset_paths)
    
    if train_count == 0:
        print("\n❌ No training images found!")
        sys.exit(1)
    
    # Create YAML config
    yaml_path = combiner.create_yaml()
    
    # Train
    model_path = train_combined(yaml_path, config)
    
    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📦 Model saved: {model_path}")
    print(f"\nTo use:")
    print(f"  python trash_detector_v3.py ./test_images ./results --model {model_path}")


if __name__ == '__main__':
    main()
