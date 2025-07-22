import sys
import argparse
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import cv2
import time
import json
from app.models.enhancer import run_model_only, run_preprocessing_only, run_full_pipeline

def glare_enhance_with_timing(input_path, output_path, mode_choices, glare_thresh):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_timings = []
    frame_count = 0
    total_processing_time = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        
        if '1' in mode_choices:
            processed = run_model_only(frame)
        elif '2' in mode_choices:
            processed = run_preprocessing_only(frame, glare_thresh)
        elif '3' in mode_choices:
            processed = run_full_pipeline(frame, glare_thresh)
        else:
            processed = cv2.resize(frame, (w, h))

        frame_processing_time = time.time() - frame_start_time
        total_processing_time += frame_processing_time

        frame_timings.append({
            "frame_number": frame_count,
            "total_time": frame_processing_time,
            "classify_time": 0,
            "process_time": frame_processing_time,
            "is_low_light": True,
            "was_enhanced": True,
            "timestamp": time.time()
        })

        processed = cv2.resize(processed, (w, h))
        out.write(processed)
        frame_count += 1

    cap.release()
    out.release()

    elapsed_time = time.time() - start_time
    timing_output_path = output_path.replace('.mp4', '_timings.json')
    timing_data = {
        "processing_method": "glare_reduction",
        "total_input_frames": frame_count,
        "total_processing_time_seconds": elapsed_time,
        "avg_time_per_frame_seconds": total_processing_time / frame_count if frame_count > 0 else 0,
        "frame_by_frame_timings": frame_timings,
        "video_info": {
            "input_path": input_path,
            "output_path": output_path,
            "original_fps": fps,
            "output_resolution": f"{w}x{h}"
        }
    }

    with open(timing_output_path, 'w') as f:
        json.dump(timing_data, f, indent=2)

    print(f"PROCESSED_COUNT={frame_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run glare enhancement")
    parser.add_argument("input_path", help="Path to input video")
    parser.add_argument("output_path", help="Path to output video")
    parser.add_argument("--mode", type=str, default="3", choices=["1", "2", "3"], help="Enhancement mode")
    parser.add_argument("--threshold", type=int, default=None, help="Optional glare threshold")

    args = parser.parse_args()

    glare_enhance_with_timing(args.input_path, args.output_path, mode_choices=[args.mode], glare_thresh=args.threshold)
