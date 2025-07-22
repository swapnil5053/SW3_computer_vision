import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import sys
import subprocess
from tqdm import tqdm
from PIL import Image
import time
import json

class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))
        if k.size() != x.size():
            raise Exception("k, haze image are different size!")
        output = k * x - k + self.b
        return F.relu(output)

def load_model(ckpt_path, device):
    net = AODnet().to(device)
    
    # Load the entire training checkpoint.
    checkpoint = torch.load(ckpt_path, map_location=device, encoding='latin1')
    
    # The actual model parameters are stored under the 'state_dict' key.
    state_dict = checkpoint['state_dict']
    
    # The model was saved with DataParallel, which adds a 'module.' prefix to keys.
    # We need to remove this prefix to match the local model structure.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove `module.`
        new_state_dict[name] = v
        
    net.load_state_dict(new_state_dict)
    net.eval()
    print(f"Model loaded from {ckpt_path}")
    return net

def process_video(input_path, output_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = load_model(model_path, device)
    
    video_capture = cv2.VideoCapture(input_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    
    # Use a temporary path for the video without audio
    temp_video_path = f"{output_path}.tmp.mp4"
    video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_timings = []
    frame_count = 0
    total_processing_time = 0
    start_time = time.time()

    for _ in tqdm(range(total_frames), desc=f"Dehazing Video"):
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_start_time = time.time()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            dehazed_tensor = network(input_tensor)
            
        dehazed_frame = np.transpose(dehazed_tensor.squeeze(0).cpu().numpy(), (1, 2, 0))
        dehazed_frame_bgr = cv2.cvtColor((dehazed_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        video_writer.write(dehazed_frame_bgr)

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
        
        frame_count += 1
        
    video_capture.release()
    video_writer.release()

    elapsed_time = time.time() - start_time
    timing_output_path = output_path.replace('.mp4', '_timings.json')
    timing_data = {
        "processing_method": "dehazing",
        "total_input_frames": frame_count,
        "total_processing_time_seconds": elapsed_time,
        "avg_time_per_frame_seconds": total_processing_time / frame_count if frame_count > 0 else 0,
        "frame_by_frame_timings": frame_timings,
        "video_info": {
            "input_path": input_path,
            "output_path": output_path,
            "original_fps": fps,
            "output_resolution": f"{width}x{height}"
        }
    }

    with open(timing_output_path, 'w') as f:
        json.dump(timing_data, f, indent=2)

    # Add audio from original video to the processed video
    command = [
        'ffmpeg', '-y', '-i', temp_video_path, '-i', input_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',  # Re-encode video to H.264 for browser compatibility
        '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0?',
        output_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python run_dehaze.py <input_video_path> <output_video_path>")
        sys.exit(1)
        
    input_video_path = sys.argv[1]
    output_video_path = sys.argv[2]
    model_path = os.path.join(os.path.dirname(__file__), "NYU_Final_AOD_38.pkl")
    
    process_video(input_video_path, output_video_path, model_path)