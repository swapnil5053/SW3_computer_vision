import cv2
import numpy as np
import time
import sys
import os

def calculate_tilt_angle(homography_matrix):
    a = homography_matrix[0, 0]
    b = homography_matrix[0, 1]
    angle_deg = np.degrees(np.arctan2(b, a))
    if angle_deg < 0:
        angle_deg += 360
    if angle_deg > 180:
        angle_deg -= 360
    return angle_deg

def determine_direction_from_matches(matches, kp1, kp2):
    if not matches:
        return "no tilt"
    x_displacements = []
    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        x_displacements.append(pt2[0] - pt1[0])
    
    if not x_displacements:
        return "no tilt"
        
    avg_x_shift = np.mean(x_displacements)
    if avg_x_shift > 1.0:
        return "right tilt"
    elif avg_x_shift < -1.0:
        return "left tilt"
    else:
        return "no tilt"

def process_video_with_overlay(video_path, output_path="output.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading the first frame of the video")
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_video_path = f"{output_path}.tmp.mp4"
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    
    total_tilt = 0
    
    CAMERA_CALIB_ANGLE = 32.46
    CORRECTION_FACTOR = 1.0 / np.sin(np.radians(CAMERA_CALIB_ANGLE))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)

        direction = "no tilt"

        if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 4:
                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                points2 = np.zeros((len(matches), 2), dtype=np.float32)

                for i, match in enumerate(matches):
                    points1[i, :] = kp1[match.queryIdx].pt
                    points2[i, :] = kp2[match.trainIdx].pt

                h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

                if h is not None:
                    tilt_angle = calculate_tilt_angle(h)
                    relative_tilt = tilt_angle * CORRECTION_FACTOR

                    if relative_tilt < -0.55 or relative_tilt > 0.0:
                        total_tilt += relative_tilt

                    direction = determine_direction_from_matches(matches, kp1, kp2)

        overlay_text = f"Total: {total_tilt:.2f} deg | Dir: {direction}"
        cv2.putText(frame, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)

        prev_gray = gray.copy()

    cap.release()
    out.release()

    # Add audio from original video to the processed video and re-encode to H.264
    command = [
        'ffmpeg', '-y', '-i', temp_video_path, '-i', video_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',  # Re-encode video to H.264
        '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0?',
        output_path
    ]
    
    try:
        import subprocess
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    print(f"Output video saved as {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python run_tilt.py <input_video_path> <output_video_path>")
        sys.exit(1)
        
    input_video_path = sys.argv[1]
    output_video_path = sys.argv[2]
    
    process_video_with_overlay(input_video_path, output_video_path)