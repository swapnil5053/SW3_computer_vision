import cv2
import numpy as np
import time

def apply_smart_selective_CLAHE_frame(
    frame,
    grid_size=(3, 3),
    percentile_thresh=110,
    std_thresh=40,
    alpha=0.7,
    use_blur=True,
    use_denoising=True
):
    img = cv2.resize(frame, (512, 512))

    if use_blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    h, w = l.shape
    gh, gw = h // grid_size[0], w // grid_size[1]

    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(4, 4))
    enhanced_l = np.copy(l)

    for i in range(0, h, gh):
        for j in range(0, w, gw):
            y1, y2 = i, min(i + gh, h)
            x1, x2 = j, min(j + gw, w)
            patch = l[y1:y2, x1:x2]

            p75 = np.percentile(patch, 75)
            std = np.std(patch)

            if p75 < percentile_thresh and std < std_thresh:
                enhanced_patch = clahe.apply(patch)

                if use_denoising:
                    enhanced_patch = cv2.fastNlMeansDenoising(enhanced_patch, None, 10, 7, 21)

                blended_patch = cv2.addWeighted(patch, 1 - alpha, enhanced_patch, alpha, 0)
                enhanced_l[y1:y2, x1:x2] = blended_patch

    merged = cv2.merge((enhanced_l, a, b))
    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_img

# 🎬 Input/output config
input_video_path = "test_video_1.mp4"
output_video_path = "side_by_side_output_skipped.avi"
frame_skip = 2  # Process every 2nd frame

cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (1024, 512))

frame_count = 0
processed_count = 0
total_processing_time = 0.0
last_combined = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        start_time = time.time()

        original_resized = cv2.resize(frame, (512, 512))
        enhanced_frame = apply_smart_selective_CLAHE_frame(frame)
        combined = np.hstack((original_resized, enhanced_frame))

        total_processing_time += (time.time() - start_time)
        last_combined = combined
        processed_count += 1

    # Reuse last processed frame for skipped frames
    if last_combined is not None:
        out.write(last_combined)

    frame_count += 1

cap.release()
out.release()

avg_time = total_processing_time / processed_count if processed_count else 0
print(f"✅ Processed {processed_count} frames (out of {frame_count})")
print(f"📁 Output saved to: {output_video_path}")
print(f"⏱️ Avg processing time per processed frame: {avg_time:.3f} sec")
