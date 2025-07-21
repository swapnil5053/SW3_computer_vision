import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir=None, frame_rate=None, start_time=0, end_time=None, use_raindrop_dataset=True):
    """
    Extract frames from a video file.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames (default: RaindropRmv/dataset)
        frame_rate (int): Extract every nth frame (None = extract all frames)
        start_time (float): Start time in seconds (default: 0)
        end_time (float): End time in seconds (None = until end of video)
        use_raindrop_dataset (bool): Whether to use RaindropRmv/dataset as default output (default: True)
    """

    if output_dir is None and use_raindrop_dataset:
        # Get the correct path to RaindropRmv/dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(os.path.dirname(current_dir))
        output_dir = os.path.join(workspace_dir, "RaindropRmv", "dataset")
    elif output_dir is None:
        output_dir = "frames"
    

    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Calculate start and end frames
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else total_frames
    
    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    saved_count = 0
    
    print(f"\nExtracting frames from {start_time}s to {end_time if end_time else 'end'}s...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret or (start_frame + frame_count) >= end_frame:
            break
        
        # Extract frame based on frame_rate parameter
        if frame_rate is None or frame_count % frame_rate == 0:
            # Generate filename as <frame_no>_B.png
            filename = f"{saved_count}_B.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save frame
            cv2.imwrite(filepath, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"  Saved {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nCompleted! Extracted {saved_count} frames to {output_dir}")
    return True


def process_video_with_model(video_path, output_dir=None, frame_rate=None, start_time=0, end_time=None):
    """
    Process video directly with ARD-CNN model without saving frames to disk first.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save predictions (default: RaindropRmv/output)
        frame_rate (int): Extract every nth frame (None = extract all frames)
        start_time (float): Start time in seconds (default: 0)
        end_time (float): End time in seconds (None = until end of video)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import here to avoid import issues when this module is used elsewhere
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(os.path.dirname(current_dir))
        raindrop_dir = os.path.join(workspace_dir, "RaindropRmv")
        sys.path.append(raindrop_dir)
        
        from removal.test_ardcnn import process_video_direct
        
        if output_dir is None:
            output_dir = os.path.join(workspace_dir, "RaindropRmv", "output")
        
        print(f"Processing video with model: {video_path}")
        
        result = process_video_direct(
            video_path=video_path,
            output_dir=output_dir,
            frame_rate=frame_rate,
            start_time=start_time,
            end_time=end_time
        )
        
        return True
        
    except Exception as e:
        print(f"Error processing video with model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("-o", "--output", 
                       help="Output directory for frames (default: RaindropRmv/dataset)")
    parser.add_argument("-r", "--rate", type=int, 
                       help="Extract every nth frame (e.g., 30 for every 30th frame)")
    parser.add_argument("-s", "--start", type=float, default=0, 
                       help="Start time in seconds (default: 0)")
    parser.add_argument("-e", "--end", type=float, 
                       help="End time in seconds (default: end of video)")
    parser.add_argument("--fps", action="store_true", 
                       help="Show video FPS and exit")
    parser.add_argument("--no-raindrop-dataset", action="store_true",
                       help="Don't use RaindropRmv/dataset as default output directory")
    parser.add_argument("--process-with-model", action="store_true",
                       help="Process video directly with ARD-CNN model")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file {args.video_path} not found")
        return
    
    # If only showing FPS
    if args.fps:
        cap = cv2.VideoCapture(args.video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            print(f"Video FPS: {fps}")
            print(f"Total frames: {total_frames}")
            print(f"Duration: {duration:.2f} seconds")
            cap.release()
        return
    
    # Extract frames or process with model
    if args.process_with_model:
        success = process_video_with_model(
            video_path=args.video_path,
            output_dir=args.output,
            frame_rate=args.rate,
            start_time=args.start,
            end_time=args.end
        )
    else:
        success = extract_frames(
            video_path=args.video_path,
            output_dir=args.output,
            frame_rate=args.rate,
            start_time=args.start,
            end_time=args.end,
            use_raindrop_dataset=not args.no_raindrop_dataset
        )
    
    if success:
        print(f"\nUsage examples:")
        print(f"  All frames: python video.py {args.video_path}")
        print(f"  Every 30th frame: python video.py {args.video_path} -r 30")
        print(f"  Time range: python video.py {args.video_path} -s 10 -e 60")
        print(f"  Custom output: python video.py {args.video_path} -o my_frames")
        print(f"  Process with model: python video.py {args.video_path} --process-with-model")


if __name__ == "__main__":
    main()