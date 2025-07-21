import cv2
import os
import re
import shutil
from pathlib import Path
dll_path = os.path.abspath("H264")
os.add_dll_directory(dll_path)

def extract_frame_number(filename):
    """Extract frame number from filename like 'frame_X_clean_frame_X.png'"""
    match = re.search(r'frame_(\d+)_clean', filename)
    return int(match.group(1)) if match else 0

def rename_frames_sequential(input_folder, temp_folder="temp_frames"):
    """Rename frames to sequential format for easier processing"""
    
    # Create temp folder
    os.makedirs(temp_folder, exist_ok=True)
    
    # Get all PNG files
    frame_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    # Sort by frame number
    frame_files.sort(key=extract_frame_number)
    
    # Rename to sequential format
    for i, filename in enumerate(frame_files):
        src = os.path.join(input_folder, filename)
        dst = os.path.join(temp_folder, f"frame_{i:06d}.png")
        shutil.copy2(src, dst)
    
    return temp_folder, len(frame_files)

def create_video_from_sequential_frames(frames_folder, output_video, fps=30):
    """Create video from sequentially named frames"""
    
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.png')])
    
    if not frame_files:
        print("No frames found!")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, layers = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"Creating video with {len(frame_files)} frames...")
    
    for i, filename in enumerate(frame_files):
        frame_path = os.path.join(frames_folder, filename)
        frame = cv2.imread(frame_path)
        
        if frame is not None:
            video_writer.write(frame)
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(frame_files)} frames")
    
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    input_folder = "lama/output"
    output_video = "cleaned_video.mp4"
    
    # Rename frames sequentially
    temp_folder, frame_count = rename_frames_sequential(input_folder)
    print(f"Renamed {frame_count} frames")
    
    # Create video
    create_video_from_sequential_frames(temp_folder, output_video, fps=30)
    
    # Clean up temp folder
    shutil.rmtree(temp_folder)
    print("Temporary files cleaned up")