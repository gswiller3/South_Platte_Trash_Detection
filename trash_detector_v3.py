#!/usr/bin/env python3
"""
South Platte River Trash Detection System v3.0
===============================================
Uses custom YOLOv8 model trained on TACO dataset.

This version detects:
- Plastic bags, wrappers, film
- Bottles, cans, cups
- Styrofoam, cartons, containers
- Cigarettes, straws, utensils
- Paper, cardboard, foil

Usage:
    python trash_detector_v3.py ./test_images ./results
    python trash_detector_v3.py ./test_images ./results --model ./trained_models/taco_trash_best.pt

Author: South Platte Capstone Team
Version: 3.0 - Custom TACO Model
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_deps():
    required = {'ultralytics': 'ultralytics', 'cv2': 'opencv-python', 'numpy': 'numpy', 'tqdm': 'tqdm'}
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing + ['-q'])
        print("Installed dependencies. Please restart.")
        sys.exit(0)

check_deps()

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Default model path (after training)
    DEFAULT_MODEL = "./trained_models/taco_trash_best.pt"
    FALLBACK_MODEL = "yolov8m.pt"  # Use if custom model not found
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    IMAGE_SIZE = 640
    
    # Size filters
    MIN_BOX_AREA = 500
    MAX_BOX_RATIO = 0.5  # Max 50% of image
    
    # Visual
    BOX_COLOR = (0, 255, 0)  # Lime green
    HIGH_CONF_COLOR = (0, 255, 255)  # Yellow for high confidence
    BOX_THICKNESS = 3
    FONT_SCALE = 0.6
    
    # Classes from our TACO training
    TACO_CLASSES = [
        'plastic_bag', 'wrapper', 'bottle', 'bottle_cap', 'can',
        'cup', 'styrofoam', 'container', 'carton', 'straw',
        'utensil', 'paper', 'cardboard', 'cigarette', 'lid',
        'plastic_film', 'foil', 'rope', 'plastic_other', 'other'
    ]
    
    # High priority classes (definitely trash, boost visibility)
    PRIORITY_CLASSES = {
        'plastic_bag', 'wrapper', 'bottle', 'can', 'cup', 
        'styrofoam', 'plastic_film', 'straw'
    }


# ============================================================================
# DETECTOR
# ============================================================================

class TACOTrashDetector:
    """Trash detector using custom TACO-trained YOLOv8 model."""
    
    def __init__(self, model_path: str = None, config: Config = None):
        self.config = config or Config()
        self.model_path = model_path or self.config.DEFAULT_MODEL
        self.model = None
        self.using_custom = False
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        model_path = Path(self.model_path)
        
        if model_path.exists():
            logger.info(f"Loading custom TACO model: {model_path}")
            self.model = YOLO(str(model_path))
            self.using_custom = True
        else:
            logger.warning(f"Custom model not found: {model_path}")
            logger.warning(f"Using fallback model: {self.config.FALLBACK_MODEL}")
            logger.warning("Run train_taco_model.py first to train custom model.")
            self.model = YOLO(self.config.FALLBACK_MODEL)
            self.using_custom = False
        
        logger.info("Model loaded.")
    
    def detect(self, image_path: str) -> Dict:
        """Run detection on single image."""
        image_path = str(image_path)
        
        # Get image dimensions
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read: {image_path}")
        img_h, img_w = img.shape[:2]
        img_area = img_w * img_h
        
        # Run inference
        results = self.model(
            image_path,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            imgsz=self.config.IMAGE_SIZE,
            verbose=False
        )[0]
        
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                # Get class name
                if self.using_custom:
                    class_name = self.config.TACO_CLASSES[cls_id] if cls_id < len(self.config.TACO_CLASSES) else 'unknown'
                else:
                    class_name = self.model.names[cls_id]
                
                # Size filtering
                w = box[2] - box[0]
                h = box[3] - box[1]
                area = w * h
                
                if area < self.config.MIN_BOX_AREA:
                    continue
                if area > img_area * self.config.MAX_BOX_RATIO:
                    continue
                
                is_priority = class_name in self.config.PRIORITY_CLASSES
                
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'priority': is_priority,
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3]),
                        'width': float(w),
                        'height': float(h)
                    }
                })
        
        return {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'has_trash': len(detections) > 0,
            'detection_count': len(detections),
            'detections': detections,
            'model': 'taco_custom' if self.using_custom else 'coco_fallback',
            'timestamp': datetime.now().isoformat()
        }
    
    def annotate(self, image_path: str, detections: List[Dict], output_path: str = None) -> np.ndarray:
        """Draw boxes on image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read: {image_path}")
        
        for det in detections:
            bbox = det['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Yellow for priority/high-conf, green otherwise
            color = self.config.HIGH_CONF_COLOR if det.get('priority') or det['confidence'] > 0.5 else self.config.BOX_COLOR
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, self.config.BOX_THICKNESS)
            
            label = f"{det['class']} {det['confidence']:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, 2)
            cv2.rectangle(img, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)
            cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, (0,0,0), 2)
        
        if output_path:
            cv2.imwrite(output_path, img)
        
        return img


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchProcessor:
    def __init__(self, detector: TACOTrashDetector):
        self.detector = detector
    
    def process_folder(self, input_path: str, output_path: str) -> Dict:
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        
        if not images:
            logger.warning(f"No images in {input_path}")
            return {}
        
        logger.info(f"Found {len(images)} images")
        
        output_path.mkdir(parents=True, exist_ok=True)
        annotated_dir = output_path / 'annotated'
        annotated_dir.mkdir(exist_ok=True)
        
        all_results = []
        trash_count = 0
        total_detections = 0
        
        for img_path in tqdm(images, desc="Processing"):
            try:
                result = self.detector.detect(str(img_path))
                all_results.append(result)
                
                if result['has_trash']:
                    trash_count += 1
                    total_detections += result['detection_count']
                    
                    out_path = annotated_dir / f"detected_{img_path.name}"
                    self.detector.annotate(str(img_path), result['detections'], str(out_path))
                    
            except Exception as e:
                logger.error(f"Error on {img_path}: {e}")
        
        # Summary
        summary = {
            'total_images': len(images),
            'images_with_trash': trash_count,
            'total_detections': total_detections,
            'detection_rate': f"{trash_count/len(images)*100:.1f}%",
            'model_used': self.detector.model_path,
            'using_custom_model': self.detector.using_custom,
            'results': all_results
        }
        
        with open(output_path / 'detections.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Class breakdown
        class_counts = {}
        for r in all_results:
            for d in r['detections']:
                c = d['class']
                class_counts[c] = class_counts.get(c, 0) + 1
        
        with open(output_path / 'summary.txt', 'w') as f:
            f.write("SOUTH PLATTE TRASH DETECTION v3.0 RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {'Custom TACO' if self.detector.using_custom else 'COCO Fallback'}\n")
            f.write(f"Total images: {len(images)}\n")
            f.write(f"Images with trash: {trash_count}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Detection rate: {summary['detection_rate']}\n\n")
            f.write("Detections by class:\n")
            for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {cls}: {cnt}\n")
        
        logger.info(f"\nResults: {trash_count}/{len(images)} images with trash")
        logger.info(f"Total detections: {total_detections}")
        
        return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='South Platte Trash Detection v3.0')
    parser.add_argument('input', help='Input folder')
    parser.add_argument('output', help='Output folder')
    parser.add_argument('--model', '-m', default=Config.DEFAULT_MODEL, help='Path to trained model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    config = Config()
    config.CONFIDENCE_THRESHOLD = args.conf
    
    print("\n" + "="*60)
    print("SOUTH PLATTE TRASH DETECTION v3.0")
    print("Custom TACO-Trained Model")
    print("="*60 + "\n")
    
    detector = TACOTrashDetector(model_path=args.model, config=config)
    processor = BatchProcessor(detector)
    
    results = processor.process_folder(args.input, args.output)
    
    print("\n" + "="*60)
    print("✅ DETECTION COMPLETE")
    print("="*60)
    print(f"\nAnnotated images: {args.output}/annotated/")
    print(f"Full results: {args.output}/detections.json")
    
    if not detector.using_custom:
        print("\n⚠️  Running with fallback model (limited accuracy)")
        print("   Run train_taco_model.py to train custom model")


if __name__ == '__main__':
    main()
