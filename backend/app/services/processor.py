import subprocess
import os
import time
import cv2
import psutil
import GPUtil
from typing import List, Optional, Tuple

METHOD = "Low Light Enhancement"
from app.utils.db_logger import log_run

def get_system_usage() -> Tuple[float, Optional[float]]:
    cpu = psutil.cpu_percent()
    try:
        gpus = GPUtil.getGPUs()
        gpu = gpus[0].load * 100 if gpus else None
    except Exception:
        gpu = None
    return cpu, gpu

def get_video_metadata(video_path: str) -> Tuple[float, str, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, "Unknown", 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0
    resolution = f"{width}x{height}"

    cap.release()
    return duration, resolution, fps, frame_count

def process_video_file(input_path: str, method: str, user: str = "unknown_user", args: Optional[List[str]] = None) -> str:
    filename = os.path.basename(input_path)
    method = method.lower()

    valid_methods = {
        "clahe": ("CLAHE", ["python", "app/models/run_clahe.py"]),
        "unet": ("UNet", ["python", "app/models/run_unet.py"]),
        "unet_selective": ("UNet Selective", ["python", "app/models/run_unet.py", "--selective"]),
        "flare-reduction": ("Flare Reduction", ["python", "app/models/run_glare.py"]),
        "glare-dim": ("Glare Dimming", ["python", "app/models/run_glare.py"]),
        "combined": ("Combined Enhancement", ["python", "app/models/run_glare.py"]),
        "dehazing": ("Dehazing", ["python", "app/models/run_dehaze.py"]),
        "tilt": ("Tilt Correction", ["python", "app/models/run_tilt.py"])
    }

    if method in valid_methods:
        model_used, base_cmd = valid_methods[method]
    else:
        print(f"[ERROR] Invalid method '{method}': falling back to UNet.")
        method = "unet"
        model_used, base_cmd = valid_methods["unet"]

    output_path = f"processed_videos/processed_{method}_{filename}"

    if method in ["flare-reduction", "glare-dim", "combined"]:
        glare_mode = {
            "flare-reduction": "1",
            "glare-dim": "2",
            "combined": "3"
        }[method]
        cmd = base_cmd + [input_path, "--mode", glare_mode]
        if args:
            cmd += args
        cmd += [output_path]
    else:
        cmd = base_cmd + [input_path, output_path]
        if args:
            cmd += args

    print(f"[INFO] Running command: {' '.join(cmd)}")

    error = None
    processed_count = 0

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()

        for line in result.stdout.splitlines():
            if line.startswith("PROCESSED_COUNT="):
                processed_count = int(line.split("=")[1])
                break

        if os.path.exists(output_path):
            total_time = end_time - start_time
            duration, resolution, fps, frame_count = get_video_metadata(input_path)
            avg_delay_per_frame = total_time / frame_count if frame_count else 0
            cpu_usage, gpu_usage = get_system_usage()
            device = "GPU" if gpu_usage is not None else "CPU"

            log_run(
                process=METHOD,
                model_used=model_used,
                video_duration=duration,
                resolution=resolution,
                fps=fps,
                total_frames=frame_count,
                processed_frames=processed_count,
                total_time=total_time,
                avg_delay_per_frame=avg_delay_per_frame,
                device=device,
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                input_file=input_path,
                output_file=output_path,
                error=error
            )
            return output_path
        else:
            error = "Output file missing."
            print(f"[ERROR] Output video not found at: {output_path}")

    except subprocess.CalledProcessError as e:
        error = str(e)
        print(f"[ERROR] Video processing failed: {error}")
        print(f"[STDERR] {e.stderr}")
        print(f"[STDOUT] {e.stdout}")

    duration, resolution, fps, frame_count = get_video_metadata(input_path)
    cpu_usage, gpu_usage = get_system_usage()
    device = "GPU" if gpu_usage is not None else "CPU"

    log_run(
        process=METHOD,
        model_used=model_used,
        video_duration=duration,
        resolution=resolution,
        fps=fps,
        total_frames=frame_count,
        processed_frames=0,
        total_time=0,
        avg_delay_per_frame=0,
        device=device,
        cpu_usage=cpu_usage,
        gpu_usage=gpu_usage,
        input_file=input_path,
        output_file=output_path,
        error=error
    )

    return ""
