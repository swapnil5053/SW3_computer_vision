from datetime import datetime
from pymongo import MongoClient
import platform
import psutil
import GPUtil


# === MongoDB Connection ===
client = MongoClient("mongodb://localhost:27017")  # change URI if needed
db = client["video_processing_logs"]
logs_collection = db["processing_logs"]


def get_device_specs():
    """Returns basic CPU and GPU model names."""
    cpu = platform.processor() or platform.machine()

    gpus = GPUtil.getGPUs()
    gpu = gpus[0].name if gpus else "None"

    return cpu, gpu


def log_run(
    process: str,                     # e.g. "low-light enhancement", "rain removal"
    model_used: str,                 # e.g. "UNet", "CLAHE"
    video_duration: float,           # in seconds
    resolution: str,                 # e.g. "1920x1080"
    fps: float,
    total_frames: int,
    processed_frames: int,
    total_time: float,               # in seconds
    avg_delay_per_frame: float,      # optional, can compute from above
    device: str,                     # "CPU" or "GPU"
    cpu_usage: float,
    gpu_usage: float,
    input_file: str,
    output_file: str,
    error: str = None
):
    cpu_name, gpu_name = get_device_specs()

    log_entry = {
        "timestamp": datetime.now(),
        "process_category": process,
        "model_used": model_used,
        "video": {
            "input_file": input_file,
            "output_file": output_file,
            "duration": video_duration,
            "resolution": resolution,
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": processed_frames
        },
        "performance": {
            "total_time": total_time,
            "avg_delay_per_frame": avg_delay_per_frame,
            "cpu_usage_percent": cpu_usage,
            "gpu_usage_percent": gpu_usage,
            "device_used": device,
            "device_specs": {
                "cpu": cpu_name,
                "gpu": gpu_name
            }
        },
        "status": "error" if error else "success",
        "error_message": error
    }

    try:
        logs_collection.insert_one(log_entry)
        print("Log inserted successfully.")
    except Exception as e:
        print(f" Failed to log run: {e}")