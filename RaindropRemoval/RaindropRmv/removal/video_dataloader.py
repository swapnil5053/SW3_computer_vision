import numpy as np
import tensorflow as tf
import cv2
from typing import List, Generator, Tuple, Optional


class VideoDataLoader:
    """
    A dataloader that processes video frames directly without saving to disk.
    Compatible with the ARD-CNN model expectations.
    """
    
    def __init__(self, batch_size=32, target_size=(256, 512)):
        self.batch_size = batch_size
        self.target_size = target_size
        
    def frames_from_video(self, video_path: str, frame_rate: Optional[int] = None, 
                         start_time: float = 0, end_time: Optional[float] = None) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames directly from video without saving to disk.
        
        Args:
            video_path: Path to the video file
            frame_rate: Extract every nth frame (None = extract all frames)
            start_time: Start time in seconds
            end_time: End time in seconds (None = until end)
            
        Yields:
            numpy.ndarray: Video frame as numpy array
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate start and end frames
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps) if end_time else total_frames
            
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret or (start_frame + frame_count) >= end_frame:
                    break
                
                # Extract frame based on frame_rate parameter
                if frame_rate is None or frame_count % frame_rate == 0:
                    # Resize frame to target size
                    frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
                    # Convert BGR to RGB (if needed)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Normalize to [0, 1]
                    frame = frame.astype(np.float32) / 255.0
                    
                    yield frame
                
                frame_count += 1
                
        finally:
            cap.release()
    
    def create_dataset_from_video(self, video_path: str, frame_rate: Optional[int] = None,
                                 start_time: float = 0, end_time: Optional[float] = None) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset directly from video frames.
        
        Args:
            video_path: Path to the video file
            frame_rate: Extract every nth frame (None = extract all frames)
            start_time: Start time in seconds
            end_time: End time in seconds (None = until end)
            
        Returns:
            tf.data.Dataset: Dataset ready for model prediction
        """
        def frame_generator():
            for frame in self.frames_from_video(video_path, frame_rate, start_time, end_time):
                yield frame
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            frame_generator,
            output_signature=tf.TensorSpec(
                shape=(*self.target_size, 3), 
                dtype=tf.float32
            )
        )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_dataset_from_frames(self, frames: List[np.ndarray]) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from a list of frames.
        
        Args:
            frames: List of numpy arrays representing frames
            
        Returns:
            tf.data.Dataset: Dataset ready for model prediction
        """
        # Process frames
        processed_frames = []
        for frame in frames:
            # Resize if needed
            if frame.shape[:2] != self.target_size:
                frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
            
            # Ensure RGB format and normalize
            if frame.dtype != np.float32:
                frame = frame.astype(np.float32) / 255.0
            
            processed_frames.append(frame)
        
        # Convert to dataset
        dataset = tf.data.Dataset.from_tensor_slices(np.array(processed_frames))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


class MemoryEfficientVideoDataLoader(VideoDataLoader):
    """
    Memory-efficient version that processes frames in chunks to avoid loading
    entire video into memory at once.
    """
    
    def __init__(self, batch_size=32, target_size=(256, 512), chunk_size=100):
        super().__init__(batch_size, target_size)
        self.chunk_size = chunk_size
    
    def create_chunked_dataset_from_video(self, video_path: str, frame_rate: Optional[int] = None,
                                         start_time: float = 0, end_time: Optional[float] = None) -> Generator[tf.data.Dataset, None, None]:
        """
        Create datasets in chunks to process large videos without memory issues.
        
        Args:
            video_path: Path to the video file
            frame_rate: Extract every nth frame (None = extract all frames)
            start_time: Start time in seconds
            end_time: End time in seconds (None = until end)
            
        Yields:
            tf.data.Dataset: Dataset chunks ready for model prediction
        """
        frames_buffer = []
        
        for frame in self.frames_from_video(video_path, frame_rate, start_time, end_time):
            frames_buffer.append(frame)
            
            if len(frames_buffer) >= self.chunk_size:
                # Create dataset from current chunk
                yield self.create_dataset_from_frames(frames_buffer)
                frames_buffer = []
        
        # Process remaining frames
        if frames_buffer:
            yield self.create_dataset_from_frames(frames_buffer)


# Utility function to integrate with existing test_ardcnn.py
def get_video_dataset(video_path: str, batch_size: int = 64, target_size: Tuple[int, int] = (256, 512),
                     frame_rate: Optional[int] = None, start_time: float = 0, 
                     end_time: Optional[float] = None) -> tf.data.Dataset:
    """
    Convenience function to create a dataset from video that's compatible with existing model code.
    
    Args:
        video_path: Path to the video file
        batch_size: Batch size for the dataset
        target_size: Target size for frames (height, width)
        frame_rate: Extract every nth frame (None = extract all frames)
        start_time: Start time in seconds
        end_time: End time in seconds (None = until end)
        
    Returns:
        tf.data.Dataset: Dataset ready for model.predict()
    """
    loader = VideoDataLoader(batch_size=batch_size, target_size=target_size)
    return loader.create_dataset_from_video(video_path, frame_rate, start_time, end_time)
