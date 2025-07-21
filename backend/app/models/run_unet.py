import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from unet_model import UNet  
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import time
import sys
import os
import json
import traceback
from PIL import Image

# === Argument Parsing ===
if len(sys.argv) != 3:
    print("Usage: python run_unet_classifier.py <input_video_path> <output_video_path>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# === Resolve Paths for Docker/Local Compatibility ===
input_path = os.path.abspath(input_path)
output_path = os.path.abspath(output_path)
script_dir = os.path.dirname(__file__)
unet_model_path = os.path.join(script_dir, "unet_model_30k_v1.pth")
classifier_model_path = os.path.join(script_dir, "Classifier.pt")

# === Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
frame_skip = 4

# === Load U-Net Model ===
try:
    model = UNet().to(device)
    model.load_state_dict(torch.load(unet_model_path, map_location=device))
    model.eval()
    print("U-Net model loaded successfully.")
except Exception:
    print(f"[ERROR] Failed to load U-Net model from {unet_model_path}")
    traceback.print_exc()
    sys.exit(2)

# === Load Classifier Model ===
try:
    weights = ConvNeXt_Base_Weights.DEFAULT
    classifier = convnext_base(weights=weights)
    classifier.classifier[2] = torch.nn.Linear(classifier.classifier[2].in_features, 1)
    classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    print("Classifier model loaded successfully.")
except Exception:
    print(f"[ERROR] Failed to load classifier model from {classifier_model_path}")
    traceback.print_exc()
    sys.exit(2)


# === Helper Functions & Transforms ===
classifier_transform = transforms.Compose([
    transforms.Resize((360, 360)),
    transforms.ToTensor()
])

def classify_frame(frame):
    """Classifies a frame as low-light or not."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    input_tensor = classifier_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.sigmoid(classifier(input_tensor)).item()
    return output >= 0.5  # True = Low-light frame

unet_transform = transforms.Compose([
    transforms.ToTensor()
])

def tensor_to_image(tensor):
    """Converts a PyTorch tensor back to an OpenCV image."""
    img = tensor.detach().cpu().numpy().squeeze(0)
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV

# === Video Processing ===
try:
    # --- Open Input Video ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Couldn't open video: {input_path}")

    # === FIX 1: Get ORIGINAL video properties ===
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input video: {original_width}x{original_height} @ {original_fps:.2f} FPS")

    # === FIX 2: Use ORIGINAL FPS for the output video writer ===
    fourcc = cv2.VideoWriter_fourcc(*'H264') # Use 'mp4v' for better compatibility
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (original_width, original_height))

    # --- Processing Loop Variables ---
    frame_count = 0
    model_runs = 0
    enhanced_frames = 0
    start_time = time.time()
    frame_timings = []
    
    # This will hold the last frame that was processed to write during skipped frames
    last_processed_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Process one frame, then skip `frame_skip` frames
        if frame_count % (frame_skip + 1) == 0:
            frame_start = time.time()
            
            # --- Step 1: Classify frame ---
            classify_start = time.time()
            # Use a resized frame for classification for performance
            classify_frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            is_low_light = classify_frame(classify_frame_resized)
            classify_time = time.time() - classify_start

            # --- Step 2: Process frame with U-Net if needed ---
            process_start = time.time()
            if is_low_light:
                # For U-Net, resize to the model's expected input size (e.g., 512x512)
                input_img_resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
                # Convert BGR to RGB before creating tensor
                input_img_rgb = cv2.cvtColor(input_img_resized, cv2.COLOR_BGR2RGB)
                input_tensor = unet_transform(input_img_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    output_tensor = model(input_tensor)

                output_img_small = tensor_to_image(output_tensor)
                
                # === FIX 3: Resize the processed frame back to the ORIGINAL dimensions ===
                last_processed_frame = cv2.resize(output_img_small, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
                enhanced_frames += 1
            else:
                # If not low-light, the original frame is the "processed" frame
                last_processed_frame = frame
            
            process_time = time.time() - process_start
            
            # --- Timing and Statistics ---
            frame_processing_time = time.time() - frame_start
            frame_timings.append({
                "frame_number": frame_count,
                "total_time": frame_processing_time,
                "classify_time": classify_time,
                "process_time": process_time,
                "is_low_light": is_low_light,
                "was_enhanced": is_low_light,
                "timestamp": time.time()
            })
            model_runs += 1

        # Write the last valid processed frame to the output on EVERY iteration.
        # This ensures the output video is smooth and has the correct number of frames.
        if last_processed_frame is not None:
            out.write(last_processed_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")


    # --- Cleanup and Final Report ---
    cap.release()
    out.release()
    elapsed_time = time.time() - start_time

    timing_output_path = output_path.replace('.mp4', '_timings.json')
    timing_data = {
        "processing_method": "unet_with_classifier",
        "total_input_frames": frame_count,
        "frames_run_through_model": model_runs,
        "frames_enhanced_with_unet": enhanced_frames,
        "total_processing_time_seconds": elapsed_time,
        "avg_time_per_model_run_seconds": (elapsed_time / model_runs) if model_runs > 0 else 0,
        "frame_by_frame_timings": frame_timings,
        "video_info": {
            "input_path": input_path,
            "output_path": output_path,
            "original_fps": original_fps,
            "output_resolution": f"{original_width}x{original_height}"
        }
    }

    with open(timing_output_path, 'w') as f:
        json.dump(timing_data, f, indent=2)

    print("\nU-Net + Classifier Processing complete.")
    print(f"Input frames: {frame_count} | Model runs: {model_runs}")
    print(f"Frames enhanced with U-Net: {enhanced_frames}")
    print(f"Total time: {elapsed_time:.2f} sec | Avg per model run: {(elapsed_time / model_runs):.4f} sec")
    print(f"Output video saved to: {output_path}")
    print(f"Timing data saved to: {timing_output_path}")

except Exception:
    print("\n[ERROR] An error occurred during video processing:")
    traceback.print_exc()
    sys.exit(3)
