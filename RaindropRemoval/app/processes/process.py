#!/usr/bin/env python3
"""
Test script for the complete ARDCNN + LaMa pipeline
This demonstrates how the APIs now output normal PNG files
"""

import requests
import os
import time
import asyncio
import aiohttp
from datetime import datetime, timedelta
import shutil

async def test_detect_and_inpaint_with_file(
    video_path: str,
    start_time: int = 0,
    end_time: int = 10,
    frame_rate: int = 1,
    session_id: str = None
):
    """Process video through ARDCNN + LaMa pipeline with session tracking"""
    
    print("Testing frame-by-frame ARDCNN + LaMa pipeline...")
    
    # Start timing
    start_time_processing = time.time()
    print(f"Pipeline started at: {datetime.now().strftime('%H:%M:%S')}")
    
    # API endpoint - CHANGED TO USE FRAME-BY-FRAME ENDPOINT
    url = "http://localhost:8001/detect_and_inpaint"
    
    # Check if the video file exists locally
    if not os.path.exists(video_path):
        return {
            "status": "error",
            "message": f"Video file not found: {video_path}"
        }
    
    # Prepare file path for container
    container_video_path = f"/workspace/{os.path.basename(video_path)}"
    
    # Copy video to RaindropRmv directory for container access
    raindrop_video_path = f"./RaindropRmv/{os.path.basename(video_path)}"
    shutil.copy2(video_path, raindrop_video_path)
    
    try:
        # Use aiohttp for async requests
        async with aiohttp.ClientSession() as session:
            # Open and send the video file
            with open(raindrop_video_path, 'rb') as video_file:
                # Create multipart form data
                data = aiohttp.FormData()
                data.add_field('video_file', video_file, 
                             filename=os.path.basename(video_path),
                             content_type='video/mp4')
                data.add_field('start_time', str(start_time))
                data.add_field('end_time', str(end_time))
                data.add_field('frame_rate', str(frame_rate))
                data.add_field('lama_service_url', 'http://lama:8002')
                data.add_field('output_dir', '/workspace/lama/output')
                data.add_field('return_format', 'files')
                data.add_field('filename_suffix', '_clean')
                
                # When calling the Docker container, pass the session_id
                if session_id:
                    data.add_field('session_id', session_id)
                
                print(f"Sending video to frame-by-frame pipeline...")
                print(f"Video: {video_path} ({os.path.getsize(video_path)} bytes)")
                print(f"Parameters: start={start_time}s, end={end_time}s, frame_rate={frame_rate}")
                
                # Send request with longer timeout for frame-by-frame processing
                async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=600)) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Calculate processing time
                        elapsed_time = time.time() - start_time_processing
                        end_time_str = datetime.now().strftime('%H:%M:%S')
                        
                        print(f"\n{'='*60}")
                        print(f"FRAME-BY-FRAME PIPELINE SUCCESS!")
                        print(f"{'='*60}")
                        print(f"Status: {result.get('status', 'N/A')}")
                        print(f"Processing Mode: {result.get('processing_mode', 'N/A')}")
                        print(f"Pipeline: {result.get('pipeline', 'N/A')}")
                        print(f"Total frames processed: {result.get('total_frames_processed', 0)}")
                        print(f"Successful frames: {result.get('successful_frames', 0)}")
                        print(f"Failed frames: {result.get('failed_frames', 0)}")
                        print(f"Processing time: {elapsed_time:.2f} seconds")
                        print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
                        print(f"Finished: {end_time_str}")
                        
                        # Show individual frame results
                        if 'results' in result and result['results']:
                            print(f"\nFrame-by-frame results:")
                            for i, frame_result in enumerate(result['results'][:5]):  # Show first 5
                                status = frame_result.get('status', 'unknown')
                                frame_idx = frame_result.get('frame_index', i)
                                timestamp = frame_result.get('timestamp', 0)
                                print(f"  Frame {frame_idx}: {status} (at {timestamp:.2f}s)")
                            
                            if len(result['results']) > 5:
                                print(f"  ... and {len(result['results']) - 5} more frames")
                        
                        return {
                            "status": "success",
                            "pipeline_type": "frame_by_frame",
                            "processing_time": elapsed_time,
                            "result": result
                        }
                    else:
                        error_text = await response.text()
                        elapsed_time = time.time() - start_time_processing
                        
                        print(f"\nFrame-by-frame pipeline ERROR: {response.status}")
                        print(f"Response: {error_text}")
                        print(f"Processing time: {elapsed_time:.2f} seconds")
                        
                        return {
                            "status": "error",
                            "pipeline_type": "frame_by_frame",
                            "processing_time": elapsed_time,
                            "error": f"HTTP {response.status}: {error_text}"
                        }
                        
    except asyncio.TimeoutError:
        elapsed_time = time.time() - start_time_processing
        print(f"\nFrame-by-frame pipeline TIMEOUT after {elapsed_time:.2f} seconds")
        
        return {
            "status": "timeout",
            "pipeline_type": "frame_by_frame", 
            "processing_time": elapsed_time,
            "error": "Request timed out during frame-by-frame processing"
        }

    except Exception as e:
        elapsed_time = time.time() - start_time_processing
        print(f"\nFrame-by-frame pipeline EXCEPTION: {e}")
        
        return {
            "status": "error",
            "pipeline_type": "frame_by_frame",
            "processing_time": elapsed_time,
            "error": str(e)
        }

def test_detect_and_inpaint():
    """Original synchronous function for backward compatibility"""
    
    print("Testing complete ARDCNN + LaMa pipeline...")
    
    # Start timing
    start_time = time.time()
    print(f"Pipeline started at: {datetime.now().strftime('%H:%M:%S')}")
    
    # API endpoint
    url = "http://localhost:8001/detect_and_inpaint"
    
    # Test video path (using the existing video)
    video_path = "c:/Users/ASUS/SW3_computer_vision/output_video_rain.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    # Parameters for the pipeline
    data = {
        'start_time': 0,
        'end_time': 30,
        'frame_rate': 1,
        'lama_service_url': 'http://lama:8002',
        'output_dir': '/workspace/lama/output',
        'return_format': 'files',
        'filename_suffix': '_clean'
    }
    
    # Open video file
    with open(video_path, 'rb') as video_file:
        files = {'video_file': video_file}
        
        print("Sending request to ARDCNN + LaMa pipeline...")
        print(f"Video: {video_path}")
        print(f"Parameters: {data}")
        
        try:
            response = requests.post(url, files=files, data=data, timeout=300)
            
            # Calculate total time
            end_time = time.time()
            total_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                print("starting ")
                print(f"Status: {result['status']}")
                print(f"Message: {result['message']}")
                print(f"Original frames: {result.get('original_frames', 'N/A')}")
                print(f"Detected masks: {result.get('detected_masks', 'N/A')}")
                
                # Display timing information
                print(f"\nTiming Information:")
                print(f"Total Processing Time: {total_time:.2f} seconds")
                print(f"Total Processing Time: {str(timedelta(seconds=int(total_time)))}")
                
                if 'frames_processed' in result:
                    frames_processed = result['frames_processed']
                    print(f"Frames Processed: {frames_processed}")
                    print(f"Average Time per Frame: {total_time/frames_processed:.3f} seconds")
                    print(f"Processing Rate: {frames_processed/total_time:.2f} frames/second")
                
                if 'inpainted_results' in result:
                    lama_result = result['inpainted_results']
                    print(f"\nLaMa processing:")
                    print(f"  - Status: {lama_result.get('status', 'N/A')}")
                    print(f"  - Message: {lama_result.get('message', 'N/A')}")
                    print(f"  - Output directory: {lama_result.get('output_directory', 'N/A')}")
                    print(f"  - Files processed: {lama_result.get('total_processed', 'N/A')}")
                    
                    if 'output_files' in lama_result:
                        print(f"  - Output files created:")
                        for file_path in lama_result['output_files']:
                            print(f"    • {file_path}")
                
                print(f"\nPipeline: {result.get('pipeline', 'N/A')}")
                
            else:
                print(f"\nERROR: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            elapsed_time = time.time() - start_time
            print(f"\nRequest timed out after {elapsed_time:.2f} seconds (this is normal for video processing)")
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"\nException after {elapsed_time:.2f} seconds: {e}")



async def process_low_light_video(
    video_path: str,
    enhancement_method: str,
    start_time: int,
    end_time: int,
    frame_rate: int,
    session_id: str
):
    """Send video to low-light container for processing"""
    try:
        # Prepare video file for upload
        with open(video_path, 'rb') as video_file:
            files = {'video_file': video_file}
            data = {
                'enhancement_method': enhancement_method,
                'start_time': start_time,
                'end_time': end_time,
                'frame_rate': frame_rate,
                'session_id': session_id
            }
            
            # Fix: Use localhost:8005 (host:exposed_port) instead of low_light:8005
            # The main server runs on host, so it needs to use the exposed port
            response = requests.post(
                "http://localhost:8005/api/enhance",  # ✅ Changed from low_light:8005
                files=files,
                data=data,
                timeout=300
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Low-light service error: {response.text}")
                
    except Exception as e:
        raise Exception(f"Failed to process video: {str(e)}")