"""
South Platte River Trash Detection System v2.0
===============================================
Hybrid detection using:
1. YOLOv8 (COCO) - bottles, cups, cans
2. Roboflow TACO model - plastic bags, wrappers, film

Author: South Platte Capstone Team
Version: 2.0 - Hybrid COCO + TACO
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check and install required dependencies."""
    required = {
        'ultralytics': 'ultralytics',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'tqdm': 'tqdm',
        'roboflow': 'roboflow',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing + ['-q'])
        print("Dependencies installed. Please restart the script.")
        sys.exit(0)


check_dependencies()

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for trash detection."""
    
    # Confidence thresholds
    COCO_CONFIDENCE = 0.30      # Higher threshold for COCO (reduce false positives)
    TACO_CONFIDENCE = 0.25      # Slightly lower for TACO (catch more trash)
    
    # Minimum detection size (pixels) - filters out tiny false positives
    MIN_BOX_WIDTH = 30
    MIN_BOX_HEIGHT = 30
    MIN_BOX_AREA = 1500  # width * height must be at least this
    
    # Maximum detection size (filters out huge false positives like "the whole river is a bowl")
    MAX_BOX_RATIO = 0.4  # Detection can't be more than 40% of image
    
    # COCO classes that are ACTUALLY trash in a river context
    # This is deliberately restrictive to reduce false positives
    COCO_TRASH_CLASSES = {
        # Containers - very reliable detections
        'bottle',
        'cup', 
        'wine glass',
        'bowl',
        
        # Bags - handbag works for shopping bags, NOT backpack (rock false positives)
        'handbag',
        
        # Food items that could be litter
        'banana',
        'apple', 
        'orange',
        'sandwich',
        
        # Small items
        'cell phone',
        'umbrella',
        'frisbee',
        'sports ball',
    }
    
    # Classes to ALWAYS exclude even if detected
    EXCLUDE_CLASSES = {
        'person', 'people', 'human', 'man', 'woman', 'child',
        'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bear',
        'backpack',      # Too many rock false positives
        'snowboard',     # Leaf piles look like snowboards
        'skateboard',    # Flat rocks
        'surfboard',     # Driftwood
        'suitcase',      # Large rocks
        'chair',         # Rock formations
        'couch',         # Rock formations
        'bed',           # Sand bars
        'bench',         # Logs
        'potted plant',  # Bushes
        'tv',            # Reflections
        'laptop',        # Flat rocks
        'keyboard',      # Pebble patterns
    }
    
    # TACO classes we care about (Roboflow model)
    # These are the classes COCO misses
    TACO_TRASH_CLASSES = {
        # Plastic bags and film - COCO can't detect these
        'plastic bag',
        'plastic bag & wrapper', 
        'bag',
        'plastic film',
        'wrapper',
        'plastic wrapper',
        'plastic',
        
        # Styrofoam
        'styrofoam',
        'foam',
        'styrofoam piece',
        
        # Containers TACO knows
        'bottle',
        'plastic bottle',
        'glass bottle',
        'can',
        'aluminum can',
        'drink can',
        'tin can',
        'carton',
        'drink carton',
        'cup',
        'plastic cup',
        'paper cup',
        'lid',
        'bottle cap',
        'plastic lid',
        
        # Paper and cardboard
        'paper',
        'paper bag',
        'cardboard',
        'cigarette',
        'cigarette butt',
        
        # Other trash
        'straw',
        'plastic straw',
        'utensil',
        'plastic utensil',
        'container',
        'food container',
        'takeaway container',
        'packaging',
        'snack wrapper',
        'chip bag',
        'candy wrapper',
        
        # Generic
        'trash',
        'litter',
        'garbage',
        'debris',
        'waste',
        'rubbish',
    }
    
    # Visual settings
    COCO_BOX_COLOR = (0, 255, 0)    # Lime green for COCO detections
    TACO_BOX_COLOR = (0, 255, 255)  # Yellow for TACO detections
    BOX_THICKNESS = 3
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    
    # Processing
    IMAGE_SIZE = 1280  # Balance between detail and speed
    
    # Roboflow settings - UPDATE THESE
    ROBOFLOW_API_KEY = None  # Set this or use environment variable
    ROBOFLOW_MODEL = "taco-trash-annotations-in-context"
    ROBOFLOW_VERSION = 12


# ============================================================================
# HYBRID TRASH DETECTOR
# ============================================================================

class HybridTrashDetector:
    """
    Combines YOLOv8 (COCO) and Roboflow (TACO) for comprehensive trash detection.
    
    COCO catches: bottles, cups, some containers
    TACO catches: plastic bags, wrappers, styrofoam, cans
    """
    
    def __init__(self, config: Config = None, roboflow_api_key: str = None):
        self.config = config or Config()
        self.coco_model = None
        self.taco_model = None
        self.roboflow_api_key = roboflow_api_key or os.environ.get('ROBOFLOW_API_KEY') or self.config.ROBOFLOW_API_KEY
        
        self._load_coco_model()
        self._load_taco_model()
    
    def _load_coco_model(self):
        """Load YOLOv8 COCO model."""
        logger.info("Loading YOLOv8 COCO model...")
        self.coco_model = YOLO('yolov8m.pt')
        logger.info("COCO model loaded.")
    
    def _load_taco_model(self):
        """Load Roboflow TACO model."""
        if not self.roboflow_api_key:
            logger.warning("No Roboflow API key provided. TACO detection disabled.")
            logger.warning("Set ROBOFLOW_API_KEY environment variable or pass to constructor.")
            logger.warning("Get free key at: https://app.roboflow.com/")
            self.taco_model = None
            return
        
        try:
            from roboflow import Roboflow
            logger.info("Loading Roboflow TACO model...")
            rf = Roboflow(api_key=self.roboflow_api_key)
            project = rf.workspace("alex-hyams-cosqx").project(self.config.ROBOFLOW_MODEL)
            self.taco_model = project.version(self.config.ROBOFLOW_VERSION).model
            logger.info("TACO model loaded.")
        except Exception as e:
            logger.warning(f"Could not load Roboflow model: {e}")
            logger.warning("Continuing with COCO-only detection.")
            self.taco_model = None
    
    def detect(self, image_path: str) -> Dict:
        """
        Run hybrid detection on a single image.
        
        Returns combined results from both COCO and TACO models.
        """
        image_path = str(image_path)
        
        # Get image dimensions for size filtering
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img_height, img_width = img.shape[:2]
        img_area = img_width * img_height
        
        all_detections = []
        
        # Run COCO detection
        coco_detections = self._detect_coco(image_path, img_width, img_height, img_area)
        all_detections.extend(coco_detections)
        
        # Run TACO detection if available
        if self.taco_model:
            taco_detections = self._detect_taco(image_path, img_width, img_height, img_area)
            all_detections.extend(taco_detections)
        
        # Remove duplicates (same object detected by both models)
        all_detections = self._remove_duplicates(all_detections)
        
        return {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'image_size': {'width': img_width, 'height': img_height},
            'has_trash': len(all_detections) > 0,
            'detection_count': len(all_detections),
            'coco_count': len([d for d in all_detections if d['source'] == 'coco']),
            'taco_count': len([d for d in all_detections if d['source'] == 'taco']),
            'detections': all_detections,
            'timestamp': datetime.now().isoformat()
        }
    
    def _detect_coco(self, image_path: str, img_width: int, img_height: int, img_area: int) -> List[Dict]:
        """Run COCO detection with strict filtering."""
        detections = []
        
        results = self.coco_model(
            image_path,
            conf=self.config.COCO_CONFIDENCE,
            imgsz=self.config.IMAGE_SIZE,
            verbose=False
        )[0]
        
        if results.boxes is None:
            return detections
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            class_name = self.coco_model.names[cls_id].lower()
            
            # Skip excluded classes
            if class_name in self.config.EXCLUDE_CLASSES:
                continue
            
            # Skip if not in our trash list
            if class_name not in self.config.COCO_TRASH_CLASSES:
                continue
            
            # Size filtering
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            
            # Too small?
            if width < self.config.MIN_BOX_WIDTH or height < self.config.MIN_BOX_HEIGHT:
                continue
            if area < self.config.MIN_BOX_AREA:
                continue
            
            # Too big? (probably a false positive)
            if area > img_area * self.config.MAX_BOX_RATIO:
                continue
            
            detections.append({
                'class': class_name,
                'confidence': float(conf),
                'source': 'coco',
                'bbox': {
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3]),
                    'width': float(width),
                    'height': float(height)
                }
            })
        
        return detections
    
    def _detect_taco(self, image_path: str, img_width: int, img_height: int, img_area: int) -> List[Dict]:
        """Run TACO detection via Roboflow."""
        detections = []
        
        try:
            result = self.taco_model.predict(
                image_path,
                confidence=int(self.config.TACO_CONFIDENCE * 100)
            )
            predictions = result.json().get('predictions', [])
        except Exception as e:
            logger.warning(f"TACO detection failed: {e}")
            return detections
        
        for pred in predictions:
            class_name = pred.get('class', '').lower()
            conf = pred.get('confidence', 0)
            
            # Check if it's a trash class we care about
            is_trash = any(tc in class_name for tc in self.config.TACO_TRASH_CLASSES) or \
                       class_name in self.config.TACO_TRASH_CLASSES
            
            if not is_trash:
                continue
            
            # Get bounding box (Roboflow uses center + width/height format)
            cx = pred.get('x', 0)
            cy = pred.get('y', 0)
            w = pred.get('width', 0)
            h = pred.get('height', 0)
            
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            
            # Size filtering
            if w < self.config.MIN_BOX_WIDTH or h < self.config.MIN_BOX_HEIGHT:
                continue
            if w * h < self.config.MIN_BOX_AREA:
                continue
            if w * h > img_area * self.config.MAX_BOX_RATIO:
                continue
            
            detections.append({
                'class': class_name,
                'confidence': float(conf),
                'source': 'taco',
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'width': float(w),
                    'height': float(h)
                }
            })
        
        return detections
    
    def _remove_duplicates(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Remove duplicate detections (same object found by both models)."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (keep higher confidence ones)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for det in detections:
            is_duplicate = False
            for kept in keep:
                iou = self._calculate_iou(det['bbox'], kept['bbox'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(det)
        
        return keep
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def annotate_image(self, image_path: str, detections: List[Dict], output_path: str = None) -> np.ndarray:
        """Draw bounding boxes on image. Green for COCO, yellow for TACO."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        for det in detections:
            bbox = det['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Color based on source
            color = self.config.COCO_BOX_COLOR if det['source'] == 'coco' else self.config.TACO_BOX_COLOR
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, self.config.BOX_THICKNESS)
            
            # Label
            label = f"{det['class']} {det['confidence']:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                  self.config.FONT_SCALE, self.config.FONT_THICKNESS)
            
            # Label background
            cv2.rectangle(image, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)
            
            # Label text
            cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       self.config.FONT_SCALE, (0, 0, 0), self.config.FONT_THICKNESS)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchProcessor:
    """Process multiple images and generate reports."""
    
    def __init__(self, detector: HybridTrashDetector):
        self.detector = detector
    
    def process_folder(self, input_path: str, output_path: str) -> Dict:
        """Process all images in a folder."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Find images
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images = [f for f in input_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in extensions]
        
        if not images:
            logger.warning(f"No images found in {input_path}")
            return {}
        
        logger.info(f"Found {len(images)} images to process")
        
        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        annotated_dir = output_path / 'annotated'
        annotated_dir.mkdir(exist_ok=True)
        
        # Process each image
        all_results = []
        trash_images = 0
        total_detections = 0
        
        for img_path in tqdm(images, desc="Processing images"):
            try:
                result = self.detector.detect(str(img_path))
                all_results.append(result)
                
                if result['has_trash']:
                    trash_images += 1
                    total_detections += result['detection_count']
                    
                    # Save annotated image
                    annotated_path = annotated_dir / f"detected_{img_path.name}"
                    self.detector.annotate_image(str(img_path), result['detections'], str(annotated_path))
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        
        # Generate summary
        summary = {
            'total_images': len(images),
            'images_with_trash': trash_images,
            'total_detections': total_detections,
            'detection_rate': f"{(trash_images/len(images)*100):.1f}%",
            'results': all_results
        }
        
        # Save results
        with open(output_path / 'detections.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save summary
        with open(output_path / 'summary.txt', 'w') as f:
            f.write("SOUTH PLATTE TRASH DETECTION RESULTS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total images processed: {len(images)}\n")
            f.write(f"Images with trash: {trash_images}\n")
            f.write(f"Total trash items found: {total_detections}\n")
            f.write(f"Detection rate: {summary['detection_rate']}\n")
            f.write(f"\nTACO model: {'Enabled' if self.detector.taco_model else 'Disabled'}\n")
        
        logger.info(f"\nResults: {trash_images}/{len(images)} images with trash")
        logger.info(f"Total detections: {total_detections}")
        
        return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='South Platte River Trash Detection v2.0')
    parser.add_argument('input', help='Input image or folder')
    parser.add_argument('output', help='Output folder')
    parser.add_argument('--roboflow-key', '-k', help='Roboflow API key')
    parser.add_argument('--coco-conf', type=float, default=0.30, help='COCO confidence threshold')
    parser.add_argument('--taco-conf', type=float, default=0.25, help='TACO confidence threshold')
    parser.add_argument('--no-taco', action='store_true', help='Disable TACO model (COCO only)')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.COCO_CONFIDENCE = args.coco_conf
    config.TACO_CONFIDENCE = args.taco_conf
    
    # Get API key
    api_key = args.roboflow_key if not args.no_taco else None
    
    # Initialize detector
    print("\n" + "="*60)
    print("SOUTH PLATTE RIVER TRASH DETECTION SYSTEM v2.0")
    print("="*60 + "\n")
    
    detector = HybridTrashDetector(config=config, roboflow_api_key=api_key)
    
    # Process
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        result = detector.detect(str(input_path))
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if result['has_trash']:
            annotated_path = output_path / f"detected_{input_path.name}"
            detector.annotate_image(str(input_path), result['detections'], str(annotated_path))
            print(f"Found {result['detection_count']} trash items!")
        else:
            print("No trash detected.")
            
    else:
        # Folder
        processor = BatchProcessor(detector)
        processor.process_folder(str(input_path), args.output)
    
    print("\n✅ Detection complete!")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
