#!/usr/bin/env python3
"""
QUICK START - South Platte River Trash Detection
=================================================
Simple script to run trash detection on your trail camera images.

Usage:
    python quickstart.py /path/to/your/images /path/to/output

    Example with SD card:
    python quickstart.py E:/DCIM ./results

    Example with folder:
    python quickstart.py ./camera_images ./detection_results

This uses a general YOLOv8 model which can detect many object types.
For best results on river trash specifically, consider training a 
custom model using train_custom_model.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trash_detector import TrashDetector, BatchProcessor, Config

def main():
    if len(sys.argv) < 3:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║       SOUTH PLATTE RIVER TRASH DETECTION - QUICK START           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Usage:                                                          ║
║      python quickstart.py <input_folder> <output_folder>         ║
║                                                                  ║
║  Examples:                                                       ║
║      python quickstart.py E:/DCIM ./results                      ║
║      python quickstart.py /media/sdcard/DCIM ./results           ║
║      python quickstart.py ./test_images ./output                 ║
║                                                                  ║
║  Optional arguments:                                             ║
║      --max N         Process only first N images                 ║
║      --conf 0.3      Set confidence threshold (0-1)              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    # Parse optional args
    max_images = None
    conf_threshold = 0.15  # Lower threshold to catch subtle trash
    
    for i, arg in enumerate(sys.argv):
        if arg == '--max' and i + 1 < len(sys.argv):
            max_images = int(sys.argv[i + 1])
        elif arg == '--conf' and i + 1 < len(sys.argv):
            conf_threshold = float(sys.argv[i + 1])
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       SOUTH PLATTE RIVER TRASH DETECTION SYSTEM                  ║
║                    Daylight Version 1.0                          ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Confidence:    {conf_threshold}")
    if max_images:
        print(f"Max images:    {max_images}")
    print("-" * 60)
    
    # Verify input exists
    if not os.path.exists(input_folder):
        print(f"\n❌ ERROR: Input folder does not exist: {input_folder}")
        print("   Make sure your SD card is mounted or the path is correct.")
        sys.exit(1)
    
    # Initialize with default settings
    config = Config()
    config.CONFIDENCE_THRESHOLD = conf_threshold
    config.IMAGE_SIZE = 1920  # Higher for better small object detection
    
    # Create detector
    print("\n🔄 Loading YOLOv8 model (this may take a moment on first run)...")
    detector = TrashDetector(config=config)
    
    # Process images
    print("\n🔍 Starting trash detection...\n")
    processor = BatchProcessor(detector)
    summary = processor.process_folder(input_folder, output_folder, max_images=max_images)
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                      OUTPUT FILES                                ║
╠══════════════════════════════════════════════════════════════════╣""")
    print(f"║  📁 {output_folder}/annotated/")
    print(f"║     └─ Images with lime green boxes around detected trash")
    print(f"║")
    print(f"║  📁 {output_folder}/with_trash/")
    print(f"║     └─ Original images that contained trash")
    print(f"║")
    print(f"║  📁 {output_folder}/no_trash/")
    print(f"║     └─ Original images without detected trash")
    print(f"║")
    print(f"║  📄 {output_folder}/detection_results.json")
    print(f"║     └─ Full detection data (coordinates, confidence, etc.)")
    print("""╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("✅ Detection complete!")


if __name__ == '__main__':
    main()
