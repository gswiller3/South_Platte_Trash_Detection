#!/usr/bin/env python3
"""
QUICK START - South Platte Trash Detection v2.0
================================================
Hybrid detection: YOLOv8 (COCO) + Roboflow (TACO)

SETUP:
1. Get free Roboflow API key: https://app.roboflow.com/
2. Set it: export ROBOFLOW_API_KEY="your_key_here"
3. Run: python quickstart_v2.py ./test_images ./results

Without Roboflow key, runs COCO-only mode (bottles/cups but no bags).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trash_detector_v2 import HybridTrashDetector, BatchProcessor, Config


def main():
    if len(sys.argv) < 3:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║         SOUTH PLATTE TRASH DETECTION v2.0 - QUICK START              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  SETUP (one time):                                                   ║
║    1. Go to https://app.roboflow.com/ and sign up (free)             ║
║    2. Click profile → Settings → API Key → Copy                      ║
║    3. Run: export ROBOFLOW_API_KEY="your_key_here"                   ║
║                                                                      ║
║  USAGE:                                                              ║
║    python quickstart_v2.py <input_folder> <output_folder>            ║
║                                                                      ║
║  OPTIONS:                                                            ║
║    --coco-only          Skip TACO model (faster, less accurate)      ║
║    --coco-conf 0.30     COCO confidence threshold (default 0.30)     ║
║    --taco-conf 0.25     TACO confidence threshold (default 0.25)     ║
║                                                                      ║
║  EXAMPLES:                                                           ║
║    python quickstart_v2.py ./test_images ./results                   ║
║    python quickstart_v2.py ./test_images ./results --coco-only       ║
║                                                                      ║
║  WHAT EACH MODEL DETECTS:                                            ║
║    COCO (YOLOv8): bottles, cups, wine glasses, bowls, handbags       ║
║    TACO (Roboflow): plastic bags, wrappers, cans, styrofoam, film    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    # Parse options
    coco_only = '--coco-only' in sys.argv
    coco_conf = 0.30
    taco_conf = 0.25
    
    for i, arg in enumerate(sys.argv):
        if arg == '--coco-conf' and i + 1 < len(sys.argv):
            coco_conf = float(sys.argv[i + 1])
        if arg == '--taco-conf' and i + 1 < len(sys.argv):
            taco_conf = float(sys.argv[i + 1])
    
    # Check for API key
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    
    if coco_only:
        api_key = None
        print("\n⚠️  Running in COCO-only mode (--coco-only flag)")
        print("   Will detect: bottles, cups, bowls")
        print("   Will NOT detect: plastic bags, wrappers, cans\n")
    elif not api_key:
        print("\n" + "="*60)
        print("⚠️  NO ROBOFLOW API KEY FOUND")
        print("="*60)
        print("\nRunning in COCO-only mode (limited detection).")
        print("\nTo enable full detection (bags, wrappers, cans):")
        print("  1. Sign up free: https://app.roboflow.com/")
        print("  2. Get API key: Profile → Settings → API Key")
        print("  3. Set it: export ROBOFLOW_API_KEY=\"your_key\"")
        print("  4. Re-run this script")
        print("\n" + "="*60 + "\n")
    
    # Configure
    config = Config()
    config.COCO_CONFIDENCE = coco_conf
    config.TACO_CONFIDENCE = taco_conf
    
    print("\n" + "="*60)
    print("SOUTH PLATTE RIVER TRASH DETECTION v2.0")
    print("="*60)
    print(f"\nInput:  {input_folder}")
    print(f"Output: {output_folder}")
    print(f"COCO confidence: {coco_conf}")
    print(f"TACO confidence: {taco_conf}")
    print(f"TACO model: {'Enabled' if api_key else 'Disabled'}")
    print()
    
    # Initialize and run
    detector = HybridTrashDetector(config=config, roboflow_api_key=api_key)
    processor = BatchProcessor(detector)
    
    results = processor.process_folder(input_folder, output_folder)
    
    print("\n" + "="*60)
    print("✅ DETECTION COMPLETE!")
    print("="*60)
    print(f"\nAnnotated images: {output_folder}/annotated/")
    print(f"Full results: {output_folder}/detections.json")
    print(f"Summary: {output_folder}/summary.txt")
    
    if not api_key:
        print("\n💡 TIP: Add Roboflow API key to detect plastic bags & wrappers")


if __name__ == '__main__':
    main()
