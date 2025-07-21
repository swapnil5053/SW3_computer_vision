import math
import glob
from os.path import basename
import sys
import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy import misc
from removal.ard_cnn import ARDCNN
from removal.video_dataloader import get_video_dataset, VideoDataLoader
import cv2
from sklearn import metrics
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from IPython import embed

#LEN = 2232
LEN = 5
BATCH_SIZE = 8  # Reduced from 64 to prevent OOM
STEPS = math.ceil(LEN / BATCH_SIZE)

STEPS_EVAL = math.ceil(500 / BATCH_SIZE)


def get_data(path):
    # key = None
    # if 'youtube' not in path:
    #     key = lambda x: int(basename(x).split('_')[0])
    images = glob.glob(path + '*.png')
    images.sort()
    mask = glob.glob(path + '*_M.png')
    mask.sort()

    images = np.array(images)
    mask = np.array(mask)

    if len(mask) != 0:
        data = tf.data.Dataset.from_tensor_slices((images, mask))
        data = data.map(lambda img, msk: map_func(img, msk), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        data = tf.data.Dataset.from_tensor_slices(images)
        data = data.map(lambda img: map_func(img), num_parallel_calls=tf.data.AUTOTUNE)
    
    data = data.batch(BATCH_SIZE)
    data = data.prefetch(tf.data.AUTOTUNE)

    return data

def map_func(image, mask=None):
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image)
    # Remove the fixed shape setting and add resizing
    image = tf.image.resize(image, [256, 512])  # Resize to expected dimensions
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if mask is not None:
        mask = tf.io.read_file(mask)
        mask = tf.image.decode_png(mask)
        mask = tf.image.resize(mask, [256, 512])  # Resize mask too
        mask = tf.reshape(mask, (256, 512, 1))
        mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)

        return image, mask

    return image


def get_all(input_path=None, is_video=False):
    """
    Process images from directory or video file.
    
    Args:
        input_path: Path to input (directory for images, file for video)
        is_video: Whether input is a video file
    """
    if is_video and input_path:
        # Process video directly
        return process_video_direct(input_path)
    
    # Original image processing code
    images = input_path if input_path else '/workspace/dataset/' # input directory
    try:
        dataset = get_data(images)
    except Exception as e:
        print(f"an error occurrent: {e}")

    image_input = keras.Input(shape=(256, 512, 3), name='rain')
    ard_cnn = ARDCNN(image_input, False)
    model = keras.Model(image_input, ard_cnn.outputs)

    weights = '/workspace/model/ard.40_0.00649.hdf5' # select model weights
    model.load_weights(weights)

    out = model.predict(dataset, steps=STEPS)


    key = lambda x: int(basename(x).split('_')[0])
    path = glob.glob(images + '*_B.png')
    path.sort(key=key)

    print(f"Number of predictions: {len(out)}")
    print(f"Number of image paths: {len(path)}")

    for i, img in enumerate(out):
        if i >= len(path):
            print(f"Warning: More predictions than image paths at index {i}")
            break
            
        img = np.where(img < 0.5, 0.0, 1.0)
        img = img * 255.0
        img = img.astype(np.uint8)
        
        # output directory
        original_filename = os.path.basename(path[i])
        output_filename = original_filename.replace('_B.png', '_M_pred.png')
        output_path = os.path.join('/workspace/output/', output_filename)
        
        print(f"Saving {original_filename} -> {output_filename}")
        cv2.imwrite(output_path, img)

def eval_all():
    images = 'dataset/rain_test/cityscapes_small/'
    dataset = get_data(images)

    image_input = keras.Input(shape=(256, 512, 3), name='rain')
    ard_cnn = ARDCNN(image_input, False)
    model = keras.Model(image_input, ard_cnn.outputs)

    weights = glob.glob('../model/w_ard*')
    weights.sort()

    gt = get_gt(images)
    confusion_matrixs = []

    for i, weight in enumerate(weights):
        model.load_weights(weight)
        res = model.predict(dataset, steps=STEPS_EVAL)
        res = np.where(res > 0.9, 1.0, 0.0)
        confusion_matrixs.append(metrics.confusion_matrix(gt, res.flatten()))

    show_score(confusion_matrixs)


def get_gt(path):
    gt = np.ndarray((500, 256, 512))
    path = glob.glob(path + '*_M.png')  
    path.sort()
    for i, img in enumerate(path):
        gt[i] = cv2.imread(img, 0)

    return gt.flatten() / 255

def show_score(confusion_matrixs):
    scores = []
    for i in range(len(confusion_matrixs)):
        TP = confusion_matrixs[i][1, 1]
        FP = confusion_matrixs[i][0, 1]
        TN = confusion_matrixs[i][0, 0]
        FN = confusion_matrixs[i][1, 0]

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        scores.append((i, precision, recall, f1))

    scores.sort(key=lambda x: x[-1])
    for i, p, r, f1 in scores:
        print('Index: %d, precision: %f, recall: %f, f1: %f' % (i, p, r, f1))

def get_one():
    #image = misc.imread("dataset/rain_test/youtube/geko/01_B.png")
    image = misc.imread("../result/fuse/00000.png")
    image = image / 255.0
    image = image.reshape((1, 256, 512, 3))
    image_input = keras.Input(shape=(256, 512, 3), name='rain')

    ard_cnn = ARDCNN(image_input, False)
    model = keras.Model(image_input, ard_cnn.outputs)

    weights = glob.glob('../model/ard*')
    weights.sort()

    for i, j in enumerate(weights):
        model.load_weights(j)
        out = model.predict(image)
        out = out.reshape((256, 512))
        out = np.where(out > 0.5, 1.0, 0.0)
        # Replace deprecated misc.imsave
        cv2.imwrite(f'../result/ardcnn/{i:02d}.png', (out * 255).astype(np.uint8))

def process_video_direct(video_path, output_dir='/workspace/output/', frame_rate=None, 
                        start_time=0, end_time=None, weights_path='/workspace/model/ard.40_0.00649.hdf5'):
    """
    Process video directly without saving frames to disk first.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save prediction results
        frame_rate: Extract every nth frame (None or 1 = extract all frames)
        start_time: Start time in seconds
        end_time: End time in seconds (None = until end)
        weights_path: Path to the model weights
    """
    try:
        # Get video properties for frame calculation
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Create dataset directly from video
        dataset = get_video_dataset(
            video_path=video_path,
            batch_size=BATCH_SIZE,
            target_size=(256, 512),
            frame_rate=frame_rate,
            start_time=start_time,
            end_time=end_time
        )
        
        # Create and load model
        image_input = keras.Input(shape=(256, 512, 3), name='rain')
        ard_cnn = ARDCNN(image_input, False)
        model = keras.Model(image_input, ard_cnn.outputs)
        model.load_weights(weights_path)
        
        # Get predictions
        predictions = model.predict(dataset)
        
        # Save predictions
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video name for output filenames
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"Processing {len(predictions)} frames from video: {video_name}")
        print(f"Video FPS: {fps}, Frame extraction rate: {frame_rate if frame_rate else 'all frames'}")
        
        # Calculate actual frame numbers based on extraction parameters
        start_frame = int(start_time * fps)
        frame_interval = frame_rate if frame_rate else 1
        
        for i, prediction in enumerate(predictions):
            # Convert prediction to binary mask
            mask = np.where(prediction < 0.5, 0.0, 1.0)
            mask = (mask * 255.0).astype(np.uint8)
            
            # Calculate actual frame number in the video
            actual_frame_number = start_frame + (i * frame_interval)
            
            # Calculate second and frame within that second
            second = actual_frame_number // int(fps)
            frame_in_second = actual_frame_number % int(fps)
            
            # Generate simple output filename: {second}_{frame}_mask.png
            output_filename = f"{second}_{frame_in_second}_maskXXX.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save mask
            cv2.imwrite(output_path, mask)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1} frames...")
        
        print(f"Completed! Saved {len(predictions)} predictions to {output_dir}")
        return predictions
        
    except Exception as e:
        print(f"Error processing video: {e}")
        raise


def process_video_with_original_frames(video_path, output_dir='/workspace/output/', 
                                     frame_rate=None, start_time=0, end_time=None,
                                     weights_path='/workspace/model/ard.40_0.00649.hdf5'):
    """
    Process video and save both original frames and predictions.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save results
        frame_rate: Extract every nth frame (None or 1 = extract all frames)
        start_time: Start time in seconds
        end_time: End time in seconds (None = until end)
        weights_path: Path to the model weights
    """
    try:
        # Get video properties
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calculate frame parameters
        start_frame = int(start_time * fps)
        frame_interval = frame_rate if frame_rate else 1
        
        # Create video dataloader
        loader = VideoDataLoader(batch_size=BATCH_SIZE, target_size=(256, 512))
        
        # Create and load model
        image_input = keras.Input(shape=(256, 512, 3), name='rain')
        ard_cnn = ARDCNN(image_input, False)
        model = keras.Model(image_input, ard_cnn.outputs)
        model.load_weights(weights_path)
        
        # Process frames and save both originals and predictions
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"Processing video: {video_name}")
        print(f"Video FPS: {fps}, Frame extraction rate: {frame_rate if frame_rate else 'all frames'}")
        
        frames_buffer = []
        frame_count = 0
        
        # Process frames in batches
        for frame in loader.frames_from_video(video_path, frame_rate, start_time, end_time):
            frames_buffer.append(frame)
            
            # Process batch when buffer is full
            if len(frames_buffer) >= BATCH_SIZE:
                # Create dataset and predict
                dataset = loader.create_dataset_from_frames(frames_buffer)
                predictions = model.predict(dataset, verbose=0)
                
                # Save results
                for i, (original_frame, prediction) in enumerate(zip(frames_buffer, predictions)):
                    # Calculate actual frame number in the video
                    actual_frame_number = start_frame + (frame_count * frame_interval)
                    second = actual_frame_number // int(fps)
                    frame_in_second = actual_frame_number % int(fps)
                    
                    # Save original frame: {second}_{frame}_original.png
                    original_filename = f"{second}_{frame_in_second}_original.png"
                    original_path = os.path.join(output_dir, original_filename)
                    original_uint8 = (original_frame * 255.0).astype(np.uint8)
                    original_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(original_path, original_bgr)
                    
                    # Save prediction: {second}_{frame}_mask.png
                    mask = np.where(prediction < 0.5, 0.0, 1.0)
                    mask = (mask * 255.0).astype(np.uint8)
                    pred_filename = f"{second}_{frame_in_second}_mask.png"
                    pred_path = os.path.join(output_dir, pred_filename)
                    cv2.imwrite(pred_path, mask)
                    
                    frame_count += 1
                
                if frame_count % 50 == 0:
                    print(f"  Processed {frame_count} frames...")
                
                frames_buffer = []
        
        # Process remaining frames
        if frames_buffer:
            dataset = loader.create_dataset_from_frames(frames_buffer)
            predictions = model.predict(dataset, verbose=0)
            
            for i, (original_frame, prediction) in enumerate(zip(frames_buffer, predictions)):
                # Calculate actual frame number in the video
                actual_frame_number = start_frame + (frame_count * frame_interval)
                second = actual_frame_number // int(fps)
                frame_in_second = actual_frame_number % int(fps)
                
                # Save original frame: {second}_{frame}_original.png
                original_filename = f"{second}_{frame_in_second}_original.png"
                original_path = os.path.join(output_dir, original_filename)
                original_uint8 = (original_frame * 255.0).astype(np.uint8)
                original_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(original_path, original_bgr)
                
                # Save prediction: {second}_{frame}_mask.png
                mask = np.where(prediction < 0.5, 0.0, 1.0)
                mask = (mask * 255.0).astype(np.uint8)
                pred_filename = f"{second}_{frame_in_second}_mask.png"
                pred_path = os.path.join(output_dir, pred_filename)
                cv2.imwrite(pred_path, mask)
                
                frame_count += 1
        
        print(f"Completed! Processed {frame_count} frames and saved to {output_dir}")
        return frame_count
        
    except Exception as e:
        print(f"Error processing video: {e}")
        raise