"""
South Platte River Trash Detection System - Daylight Version
============================================================
A computer vision system for detecting floating trash in river images
using YOLOv8 trained on trash/litter datasets.

Based on research from:
- TACO Dataset (Trash Annotations in Context)
- SS-YOLOv8 water surface litter detection (79.9% mAP)
- IWHR floating debris detection benchmarks

Author: South Platte Capstone Team
Version: 1.0 - Daylight Images Only
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
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


# Check dependencies before other imports
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
    """Configuration settings for the trash detection system."""
    
    # Model settings
    MODEL_SIZE = 'yolov8m.pt'  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    CONFIDENCE_THRESHOLD = 0.15  # Lower threshold to catch subtle trash
    IOU_THRESHOLD = 0.45  # IoU threshold for NMS
    
    # Classes to EXCLUDE (not trash - filter these OUT)
    EXCLUDE_CLASSES = {
        # Humans
        'person', 'people', 'human', 'man', 'woman', 'child', 'boy', 'girl',
        # Vehicles
        'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        # Large animals
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        # Small animals (could be near water)
        'bird', 'cat', 'dog',
        # Street infrastructure
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    }
    
    # Classes to INCLUDE (likely trash in river environment)
    TRASH_CLASSES = {
        'bottle', 'cup', 'bowl', 'wine glass',
        'fork', 'knife', 'spoon',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
        'hot dog', 'pizza', 'donut', 'cake',  # food waste
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    }
    
    # Visual settings
    BOX_COLOR = (0, 255, 0)  # Lime green in BGR format
    BOX_THICKNESS = 3
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    
    # Processing settings
    BATCH_SIZE = 4  # Reduced for higher resolution processing
    IMAGE_SIZE = 1920  # Higher resolution for small trash detection
    
    # Output settings
    SAVE_ANNOTATED = True
    SAVE_CROPS = False  # Save cropped detections
    SAVE_JSON = True  # Save detection results as JSON


# ============================================================================
# TRASH DETECTION ENGINE
# ============================================================================

class TrashDetector:
    """
    YOLOv8-based trash detection system optimized for river monitoring.
    """
    
    def __init__(self, model_path: str = None, config: Config = None):
        """
        Initialize the trash detector.
        
        Args:
            model_path: Path to custom trained model, or None to use pretrained
            config: Configuration object
        """
        self.config = config or Config()
        self.model = None
        self.model_path = model_path
        self._load_model()
        
    def _load_model(self):
        """Load the YOLOv8 model."""
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading custom model from {self.model_path}")
            self.model = YOLO(self.model_path)
        else:
            logger.info(f"Loading pretrained {self.config.MODEL_SIZE}")
            self.model = YOLO(self.config.MODEL_SIZE)
            
        # Log model info
        logger.info(f"Model loaded with {len(self.model.names)} classes")
        
    def detect_single(self, image_path: str) -> Dict:
        """
        Detect trash in a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing detection results
        """
        results = self.model(
            image_path,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            imgsz=self.config.IMAGE_SIZE,
            verbose=False
        )[0]
        
        return self._process_results(results, image_path)
    
    def detect_batch(self, image_paths: List[str], progress: bool = True) -> List[Dict]:
        """
        Detect trash in a batch of images.
        
        Args:
            image_paths: List of paths to image files
            progress: Show progress bar
            
        Returns:
            List of dictionaries containing detection results
        """
        all_results = []
        
        # Process in batches
        iterator = range(0, len(image_paths), self.config.BATCH_SIZE)
        if progress:
            iterator = tqdm(iterator, desc="Processing images", unit="batch")
        
        for i in iterator:
            batch_paths = image_paths[i:i + self.config.BATCH_SIZE]
            
            # Run inference on batch
            batch_results = self.model(
                batch_paths,
                conf=self.config.CONFIDENCE_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                imgsz=self.config.IMAGE_SIZE,
                verbose=False
            )
            
            # Process each result
            for result, path in zip(batch_results, batch_paths):
                all_results.append(self._process_results(result, path))
                
        return all_results
    
    def _process_results(self, results, image_path: str) -> Dict:
        """
        Process YOLO results into a structured dictionary.
        Filters out humans and non-trash objects.
        
        Args:
            results: YOLO results object
            image_path: Path to the source image
            
        Returns:
            Structured detection results
        """
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = self.model.names[cls_id].lower()
                
                # STRICT FILTER: Skip if class contains 'person' anywhere
                if 'person' in class_name or 'human' in class_name or 'people' in class_name:
                    continue
                
                # Skip if in exclude list
                if class_name in self.config.EXCLUDE_CLASSES:
                    continue
                
                # Only include if it's a known trash class
                if class_name not in self.config.TRASH_CLASSES:
                    continue
                
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3]),
                        'width': float(box[2] - box[0]),
                        'height': float(box[3] - box[1])
                    }
                })
        
        return {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'has_trash': len(detections) > 0,
            'detection_count': len(detections),
            'detections': detections,
            'timestamp': datetime.now().isoformat()
        }
    
    def annotate_image(self, image_path: str, detections: List[Dict], 
                       output_path: str = None) -> np.ndarray:
        """
        Draw lime green bounding boxes on detected trash.
        
        Args:
            image_path: Path to source image
            detections: List of detection dictionaries
            output_path: Optional path to save annotated image
            
        Returns:
            Annotated image as numpy array
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Draw each detection
        for det in detections:
            bbox = det['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Draw bounding box in lime green
            cv2.rectangle(
                image, 
                (x1, y1), (x2, y2), 
                self.config.BOX_COLOR, 
                self.config.BOX_THICKNESS
            )
            
            # Create label
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                self.config.FONT_SCALE, 
                self.config.FONT_THICKNESS
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (x1, y1 - label_height - 10),
                (x1 + label_width + 10, y1),
                self.config.BOX_COLOR,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.FONT_SCALE,
                (0, 0, 0),  # Black text
                self.config.FONT_THICKNESS
            )
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, image)
            
        return image


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchProcessor:
    """
    Process batches of images from SD card or folder.
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, detector: TrashDetector):
        """
        Initialize batch processor.
        
        Args:
            detector: TrashDetector instance
        """
        self.detector = detector
        
    def find_images(self, input_path: str, recursive: bool = True) -> List[str]:
        """
        Find all image files in a directory.
        
        Args:
            input_path: Directory path to search
            recursive: Search subdirectories
            
        Returns:
            List of image file paths
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")
        
        if input_path.is_file():
            if input_path.suffix.lower() in self.SUPPORTED_FORMATS:
                return [str(input_path)]
            else:
                raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        # Find all images
        images = []
        pattern = '**/*' if recursive else '*'
        
        for ext in self.SUPPORTED_FORMATS:
            images.extend(input_path.glob(f'{pattern}{ext}'))
            images.extend(input_path.glob(f'{pattern}{ext.upper()}'))
        
        return sorted([str(p) for p in images])
    
    def process_folder(self, input_folder: str, output_folder: str,
                       max_images: int = None) -> Dict:
        """
        Process all images in a folder.
        
        Args:
            input_folder: Folder containing images
            output_folder: Folder to save results
            max_images: Maximum number of images to process (None = all)
            
        Returns:
            Summary statistics
        """
        # Find images
        logger.info(f"Scanning {input_folder} for images...")
        image_paths = self.find_images(input_folder)
        
        if max_images:
            image_paths = image_paths[:max_images]
            
        logger.info(f"Found {len(image_paths)} images to process")
        
        if not image_paths:
            logger.warning("No images found!")
            return {'error': 'No images found'}
        
        # Create output directories
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        annotated_dir = output_path / 'annotated'
        annotated_dir.mkdir(exist_ok=True)
        
        no_trash_dir = output_path / 'no_trash'
        no_trash_dir.mkdir(exist_ok=True)
        
        with_trash_dir = output_path / 'with_trash'
        with_trash_dir.mkdir(exist_ok=True)
        
        # Process images
        logger.info("Running trash detection...")
        results = self.detector.detect_batch(image_paths, progress=True)
        
        # Organize results
        summary = {
            'total_images': len(results),
            'images_with_trash': 0,
            'images_without_trash': 0,
            'total_detections': 0,
            'detection_classes': {},
            'results': []
        }
        
        logger.info("Saving annotated images...")
        for result in tqdm(results, desc="Saving results"):
            filename = result['filename']
            image_path = result['image_path']
            
            if result['has_trash']:
                summary['images_with_trash'] += 1
                summary['total_detections'] += result['detection_count']
                
                # Count detection classes
                for det in result['detections']:
                    cls = det['class']
                    summary['detection_classes'][cls] = summary['detection_classes'].get(cls, 0) + 1
                
                # Save annotated image
                annotated_path = annotated_dir / f"annotated_{filename}"
                self.detector.annotate_image(
                    image_path, 
                    result['detections'],
                    str(annotated_path)
                )
                
                # Copy original to with_trash folder
                shutil.copy2(image_path, with_trash_dir / filename)
                
            else:
                summary['images_without_trash'] += 1
                # Copy to no_trash folder
                shutil.copy2(image_path, no_trash_dir / filename)
            
            summary['results'].append(result)
        
        # Save JSON results
        if self.detector.config.SAVE_JSON:
            json_path = output_path / 'detection_results.json'
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Results saved to {json_path}")
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print processing summary."""
        print("\n" + "="*60)
        print("TRASH DETECTION SUMMARY")
        print("="*60)
        print(f"Total images processed: {summary['total_images']}")
        print(f"Images WITH trash:      {summary['images_with_trash']} ({100*summary['images_with_trash']/summary['total_images']:.1f}%)")
        print(f"Images WITHOUT trash:   {summary['images_without_trash']} ({100*summary['images_without_trash']/summary['total_images']:.1f}%)")
        print(f"Total detections:       {summary['total_detections']}")
        
        if summary['detection_classes']:
            print("\nDetection breakdown by class:")
            for cls, count in sorted(summary['detection_classes'].items(), key=lambda x: -x[1]):
                print(f"  {cls}: {count}")
        print("="*60 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the trash detection system."""
    parser = argparse.ArgumentParser(
        description='South Platte River Trash Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a folder of images
  python trash_detector.py --input /path/to/images --output ./results
  
  # Process with custom model
  python trash_detector.py --input ./images --output ./results --model custom_weights.pt
  
  # Process only first 50 images
  python trash_detector.py --input ./images --output ./results --max 50
  
  # Use higher confidence threshold
  python trash_detector.py --input ./images --output ./results --conf 0.5
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Input folder containing images (e.g., SD card path)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output folder for results')
    parser.add_argument('--model', '-m', default=None,
                        help='Path to custom trained model weights')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--size', type=int, default=1280,
                        help='Input image size for model')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.CONFIDENCE_THRESHOLD = args.conf
    config.IMAGE_SIZE = args.size
    
    # Initialize detector
    logger.info("Initializing Trash Detection System...")
    detector = TrashDetector(model_path=args.model, config=config)
    
    # Process images
    processor = BatchProcessor(detector)
    processor.process_folder(
        input_folder=args.input,
        output_folder=args.output,
        max_images=args.max
    )
    
    logger.info("Processing complete!")


if __name__ == '__main__':
    main()
