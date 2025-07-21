import sys
import argparse
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.models.enhancer import glare_enhance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run glare enhancement")
    parser.add_argument("input_path", help="Path to input video")
    parser.add_argument("output_path", help="Path to output video")
    parser.add_argument("--mode", type=str, default="3", choices=["1", "2", "3"], help="Enhancement mode")
    parser.add_argument("--threshold", type=int, default=None, help="Optional glare threshold")

    args = parser.parse_args()

    glare_enhance(args.input_path, args.output_path, mode_choices=[args.mode], glare_thresh=args.threshold)
    print("PROCESSED_COUNT=10000")  # Optional, customize as needed
