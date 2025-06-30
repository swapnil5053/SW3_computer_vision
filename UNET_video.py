import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from unet_model import UNet
import time

#  CONFIG 
video_path = "input_video_1_mallik.mp4"
output_path = "enhanced_video_mallik_1.mp4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frame_skip = 4

# Load Model 
model = UNet().to(device)
model.load_state_dict(torch.load("unet_model_30k_v1.pth", map_location=device))
model.eval()

# Preprocessing 
transform = transforms.Compose([
    transforms.ToTensor(),
])

def tensor_to_image(tensor):
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

# Video Setup 
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(" Error: Couldn't open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps // (frame_skip + 1), (width * 2, height))

frame_index = 0
frame_count = 0

# Start Timer 
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index % (frame_skip + 1) != 0:
        frame_index += 1
        continue

    original = frame.copy()
    input_img = cv2.resize(frame, (512, 512))
    input_tensor = transform(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)[0]

    output_img = tensor_to_image(output_tensor)
    output_img = cv2.resize(output_img, (original.shape[1], original.shape[0]))

    side_by_side = np.concatenate((original, output_img), axis=1)
    out.write(side_by_side)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_count += 1
    frame_index += 1

# End Timer 
end_time = time.time()
elapsed_time = end_time - start_time

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video Frame Rate: {fps} FPS")
print(f"Video Resolution: {width} x {height}")
print(f"Processed {frame_count} frames")
print(f"Time taken: {elapsed_time:.2f} seconds ({elapsed_time/frame_count:.3f} s/frame)")