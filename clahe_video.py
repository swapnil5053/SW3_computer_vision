import cv2
import numpy as np
import time
import os




# Specify the path to your input video
video_path = "test_video_hd.mp4"  
frames_to_skip = 1  
processing_size = (640, 480)  
output_path = "side_by_side_enhanced_video.avi"


def apply_CLAHE_to_frame(frame):
    """Applies CLAHE to a single video frame while retaining color."""
    # Convert to LAB color space to separate luminance from color
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the enhanced L-channel back with the original color channels
    merged = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_img


# --- Video Processing Setup ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
else:
    # Get video properties for the output file
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Adjust output FPS based on skipped frames
    output_fps = fps / (frames_to_skip + 1)

    # The output video width will be double the processing width
    output_width = processing_size[0] * 2
    output_height = processing_size[1]

    # Define the video writer for the side-by-side video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))

    print(f"Processing video: {video_path}")
    print(f"Side-by-side video will be saved to: {output_path}")

    frame_count = 0
    processed_count = 0
    total_processing_time = 0
    video_start_time = time.time()

    # Main Processing Loop 
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Process only if the frame is not being skipped
        if frame_count % (frames_to_skip + 1) == 0:
            frame_start_time = time.time()

            # Resize original frame to the standard processing size
            original_resized = cv2.resize(frame, processing_size)

            # Enhance the resized frame
            enhanced_frame = apply_CLAHE_to_frame(original_resized)
            
            # Combine the original and enhanced frames horizontally
            side_by_side_frame = np.hstack((original_resized, enhanced_frame))

            # Write the combined frame to the output video file
            out.write(side_by_side_frame)
            
            frame_end_time = time.time()
            frame_duration = frame_end_time - frame_start_time
            total_processing_time += frame_duration
            
            print(f"Processed frame {frame_count} in {frame_duration:.4f} seconds.")
            processed_count += 1
            
        frame_count += 1

    # Final Timings and Cleanup 
    video_end_time = time.time()
    total_video_duration = video_end_time - video_start_time
    
    cap.release()
    out.release()
    
    print("\n" + "="*40)
    print(" Video processing complete.")
    print(f"Total frames processed: {processed_count}")
    print(f"Total time for video processing: {total_video_duration:.2f} seconds.")
    if processed_count > 0:
        avg_time_per_frame = total_processing_time / processed_count
        print(f"Average time per processed frame: {avg_time_per_frame:.4f} seconds.")
    print(f"Side-by-side video saved as: {output_path}")
    print("="*40)