import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from app.basicsr.archs.uformer_arch import Uformer
from app.basicsr.utils.flare_util import predict_flare_from_6_channel
import os

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Uformer(img_size=512, img_ch=3, output_ch=6).to(device)
ckpt_path = os.path.join(os.path.dirname(__file__), '../models/net_g_last.pth')
ckpt_path = os.path.abspath(ckpt_path)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt.get("params_ema") or ckpt.get("params") or ckpt)
model.eval()

# === Preprocessing ===
to_tensor = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# === Golden Section Search ===
def golden_section_search(f, a, b, tol=0.01, max_iter=100):
    gr = 0.618
    for _ in range(max_iter):
        x1 = a + (1 - gr) * (b - a)
        x2 = a + gr * (b - a)
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        if abs(b - a) < tol:
            break
    return (a + b) / 2

# === MODE 1: Model Only ===
def run_model_only(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = to_tensor(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        deflare_img, _, _ = predict_flare_from_6_channel(output, torch.tensor([2.2]).to(device))
    output_np = deflare_img.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    output_np = (output_np * 255).astype(np.uint8)
    return cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

# === MODE 2: Preprocessing Only (Glare Region Only) ===
def run_preprocessing_only(frame, glare_thresh=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if glare_thresh is None:
        def dummy_thresh_func(thresh):
            mask = (gray > thresh - 5) & (gray < thresh + 5)
            return -np.sum(mask)
        glare_thresh = int(golden_section_search(dummy_thresh_func, 200, 255))

    glare_mask = (gray >= glare_thresh).astype(np.uint8) * 255

    def smooth_binary_mask(mask, kernel_size=15, sigma=5):
        mask = (mask / 255.0).astype(np.float32)
        return np.clip(cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma), 0, 1)

    def apply_glare_dimming_only(bgr_img, glare_mask):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = v.astype(np.float32)
        alpha = smooth_binary_mask(glare_mask)
        v_dimmer = v * 0.4
        v_final = v_dimmer * alpha + v * (1.0 - alpha)
        return cv2.cvtColor(cv2.merge([h, s, np.clip(v_final, 0, 255).astype(np.uint8)]), cv2.COLOR_HSV2BGR)

    return apply_glare_dimming_only(frame, glare_mask)

# === MODE 3: Full Pipeline ===
def run_full_pipeline(frame, glare_thresh=None):
    return run_preprocessing_only(run_model_only(frame), glare_thresh=glare_thresh)

# === Main Video Processor ===
def glare_enhance(input_path: str, output_path: str, mode_choices=['3'], glare_thresh=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    #out_w = 512
    #out_h = 512
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print("Processing video...")
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process based on selected mode
        if '1' in mode_choices:
            processed = run_model_only(frame)
        elif '2' in mode_choices:
            processed = run_preprocessing_only(frame, glare_thresh)
        elif '3' in mode_choices:
            processed = run_full_pipeline(frame, glare_thresh)
        else:
            processed = cv2.resize(frame, (w, h))  # fallback to original

        # Resize if needed and write
        processed = cv2.resize(processed, (w, h))
        out.write(processed)

        frame_id += 1
        if frame_id % 10 == 0:
            print(f"Processed {frame_id} frames...")

    cap.release()
    out.release()
    print(f"Done! Saved output to: {output_path}")