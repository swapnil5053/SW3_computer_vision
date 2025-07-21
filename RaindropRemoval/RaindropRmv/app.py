from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from removal.test_ardcnn import get_all, process_video_direct, process_video_with_original_frames
import traceback
import os
import tempfile
import subprocess
import time
import base64
import cv2
import numpy as np
import requests
from typing import Optional, Union
from PIL import Image


# Get the host URL for callbacks, with fallbacks for different environments
def get_host_callback_url():
    """Get the appropriate host URL for Docker container to host communication"""
    base_url = os.getenv("HOST_CALLBACK_URL")
    if base_url:
        return base_url
    
    # Try different host resolution methods
    possible_hosts = [
        "http://host.docker.internal:8000",  # Docker Desktop
        "http://172.17.0.1:8000",           # Default Docker bridge gateway
        "http://192.168.1.1:8000",          
    ]
    
    return possible_hosts[0]  # Default to first option

HOST_CALLBACK_URL = get_host_callback_url()

print(f"🚀 ARDCNN container starting...")
print(f"📡 Host callback URL: {HOST_CALLBACK_URL}")


app = FastAPI(title="ARDCNN Raindrop Detection API")

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",  # Vite dev server
        "http://localhost:5173",  # React dev server
        "http://localhost:8000",  # Your main backend
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {
        "message": "ARDCNN detection service is running",
        "host_callback_url": HOST_CALLBACK_URL
    }

@app.get("/test_callback")
def test_callback():
    """Test the connection to the host callback endpoint"""
    import requests
    try:
        response = requests.get(f"{HOST_CALLBACK_URL}/", timeout=5)
        return {
            "status": "success",
            "host_url": HOST_CALLBACK_URL,
            "host_response_code": response.status_code,
            "host_response": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {
            "status": "error",
            "host_url": HOST_CALLBACK_URL,
            "error": str(e)
        }

@app.post("/detect_and_pipeline")
async def detect_and_pipeline(
    input_data: Optional[UploadFile] = File(None, description="Video file (optional)"),
    input_path: Optional[str] = Form(None, description="Path to video file (optional)"),
    frame_rate: Optional[int] = Form(None, description="Extract every nth frame (None = all frames)"),
    start_time: float = Form(0, description="Start time in seconds"),
    end_time: Optional[float] = Form(None, description="End time in seconds"),
    lama_service_url: str = Form("http://lama:8002", description="LaMa service URL"),
    output_video_name: str = Form("cleaned_video4.mp4", description="Output video filename")
):
    """
    FULL PIPELINE: ARDCNN detection → LaMa inpainting → Video stitching
    No intermediate I/O - frames passed directly in memory
    """
    try:
        from io import BytesIO
        
        # Handle video input
        if input_data is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                content = await input_data.read()
                temp_video.write(content)
                video_path = temp_video.name
        elif input_path is not None:
            if not os.path.exists(input_path):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Video file not found: {input_path}"}
                )
            video_path = input_path
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "No input provided. Use either input_data (file) or input_path (string)"}
            )
        
        try:
            # Initialize ARDCNN model
            from removal.video_dataloader import VideoDataLoader
            from removal.ard_cnn import ARDCNN
            from tensorflow import keras
            
            # Load ARDCNN model
            image_input = keras.Input(shape=(256, 512, 3), name='rain')
            ard_cnn = ARDCNN(image_input, False)
            model = keras.Model(image_input, ard_cnn.outputs)
            model.load_weights('/workspace/model/ard.40_0.00649.hdf5')
            
            # Create output directory for final inpainted frames
            output_frames_dir = "/workspace/output_frames"
            os.makedirs(output_frames_dir, exist_ok=True)
            
            # Extract and process frames
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            processed_count = 0
            successful_frames = 0
            
            print(f"🎥 Starting pipeline for video: {os.path.basename(video_path)}")
            print(f"📊 Video FPS: {fps}, Target frame rate: {frame_rate or 1}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply time and frame rate filtering
                current_time = frame_count / fps
                if current_time < start_time:
                    frame_count += 1
                    continue
                if end_time and current_time > end_time:
                    break
                if frame_rate and frame_count % frame_rate != 0:
                    frame_count += 1
                    continue
                
                # STEP 1: ARDCNN Detection (in memory)
                original_frame = frame.copy()
                resized_frame = cv2.resize(frame, (512, 256))
                normalized_frame = resized_frame.astype(np.float32) / 255.0
                frame_batch = np.expand_dims(normalized_frame, axis=0)
                
                prediction = model.predict(frame_batch, verbose=0)[0]
                mask = np.where(prediction < 0.5, 0, 255).astype(np.uint8)
                
                # Resize mask back to original frame size
                mask_resized = cv2.resize(mask.squeeze(), (original_frame.shape[1], original_frame.shape[0]))
                
                # STEP 2: Send to LaMa (in memory via HTTP)
                frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_buffer = BytesIO()
                frame_pil.save(frame_buffer, format='PNG')
                frame_buffer.seek(0)
                
                mask_pil = Image.fromarray(mask_resized)
                mask_buffer = BytesIO()
                mask_pil.save(mask_buffer, format='PNG')
                mask_buffer.seek(0)
                
                files = [
                    ('images', (f'frame_{processed_count}.png', frame_buffer, 'image/png')),
                    ('masks', (f'mask_{processed_count}.png', mask_buffer, 'image/png'))
                ]
                
                lama_data = {
                    'return_format': 'base64',
                    'filename_suffix': f'_clean_frame_{processed_count}',
                    'frame_index': str(processed_count)
                }
                
                # STEP 3: Call LaMa
                response = requests.post(
                    f"{lama_service_url}/inpaint",
                    files=files,
                    data=lama_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    lama_result = response.json()
                    
                    # STEP 4: Save inpainted frame locally
                    if lama_result['results'] and len(lama_result['results']) > 0:
                        result_data = lama_result['results'][0]
                        
                        if result_data['format'] == 'base64':
                            # Decode base64 and save frame
                            import base64
                            frame_data = base64.b64decode(result_data['data'])
                            frame_filename = f"frame_{processed_count:06d}_clean.png"
                            frame_path = os.path.join(output_frames_dir, frame_filename)
                            
                            with open(frame_path, 'wb') as f:
                                f.write(frame_data)
                            
                            successful_frames += 1
                            print(f" Frame {processed_count} processed and saved")
                        else:
                            print(f"⚠ Frame {processed_count} - unexpected format")
                    else:
                        print(f" Frame {processed_count} - no results from LaMa")
                else:
                    print(f" Frame {processed_count} - LaMa failed: HTTP {response.status_code}")
                
                processed_count += 1
                frame_count += 1
                
                # Progress update
                if processed_count % 10 == 0:
                    print(f"📊 Processed {processed_count} frames, {successful_frames} successful")
            
            cap.release()
            
            print(f"Video stitching: {successful_frames} successful frames")
            if successful_frames > 0:
                output_video_path = f"/workspace/{output_video_name}"
                frame_files = sorted([f for f in os.listdir(output_frames_dir) if f.endswith('.png')])
                print(f"Found {len(frame_files)} frame files for stitching")

                if frame_files:
                    # Debug: List some frame files
                    print(f"Sample frame files: {frame_files[:5]}")
                    
                    # Use ffmpeg to stitch frames with simpler, more compatible settings
                    input_pattern = os.path.join(output_frames_dir, "frame_%06d_clean.png")
                    print(f"FFmpeg input pattern: {input_pattern}")

                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-y",  # Overwrite output file if exists
                        "-framerate", str(max(1, int(fps))),
                        "-i", input_pattern,
                        "-c:v", "libx264",
                        "-profile:v", "baseline",
                        "-level", "3.1",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        "-preset", "ultrafast",
                        "-crf", "28",
                        output_video_path
                    ]
                    
                    print(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
                    
                    try:
                        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
                        print(f"FFmpeg completed successfully")
                        
                        if os.path.exists(output_video_path):
                            file_size = os.path.getsize(output_video_path)
                            print(f"Output video created: {output_video_path}, size: {file_size} bytes")
                            
                            return JSONResponse(content={
                                "status": "success",
                                "message": "Pipeline completed",
                                "total_frames_processed": processed_count,
                                "successful_frames": successful_frames,
                                "output_video": output_video_name,
                                "video_file_size": file_size
                            })
                        else:
                            print(f"Error: Output video file was not created")
                            return JSONResponse(status_code=500, content={"error": "Video file not created"})
                            
                    except subprocess.CalledProcessError as e:
                        print(f"FFmpeg error: {e}")
                        print(f"FFmpeg stderr: {e.stderr}")
                        print(f"FFmpeg stdout: {e.stdout}")
                        return JSONResponse(status_code=500, content={"error": f"Video stitching failed: {e.stderr}"})
                        
                else:
                    return JSONResponse(status_code=500, content={"error": "No frames for stitching"})
            else:
                return JSONResponse(status_code=500, content={"error": "No frames processed"})

                
        finally:
            # Clean up temp video if created
            if input_data is not None and os.path.exists(video_path):
                os.unlink(video_path)
                
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": error_message,
                "traceback": error_traceback
            }
        )

@app.post("/detect")
async def detect(
    input_data: Optional[UploadFile] = File(None, description="Video file (optional)"),
    input_path: Optional[str] = Form(None, description="Path to video file (optional)"),
    frame_rate: Optional[int] = Form(None, description="Extract every nth frame (None = all frames)"),
    start_time: float = Form(0, description="Start time in seconds"),
    end_time: Optional[float] = Form(None, description="End time in seconds"),
    output_format: str = Form("masks_only", description="Output format: 'masks_only' or 'frames_and_masks'")
):
    """
    Unified detection endpoint that handles:
    - Default dataset processing (no input)
    - Video file upload processing
    - Video path processing
    - Flexible output formats
    """
    try:
        # Case 1: No input - process default dataset
        if input_data is None and input_path is None:
            result = get_all()
            return JSONResponse(content={
                "status": "success",
                "message": "Default dataset detection completed",
                "result": result
            })
        
        # Case 2: Video file upload
        elif input_data is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                content = await input_data.read()
                temp_video.write(content)
                temp_video_path = temp_video.name
            
            try:
                if output_format == "frames_and_masks":
                    # Process with original frames + masks
                    result = process_video_with_original_frames(
                        video_path=temp_video_path,
                        output_dir='/workspace/shared/',
                        frame_rate=frame_rate or 1,
                        start_time=int(start_time),
                        end_time=int(end_time) if end_time else None
                    )
                    return JSONResponse(content={
                        "status": "success",
                        "message": "Video processed with frames and masks",
                        "frames_processed": result,
                        "output_directory": "/workspace/shared/",
                        "format": "original_frames + masks for LaMa"
                    })
                else:
                    # Process masks only
                    result = process_video_direct(
                        video_path=temp_video_path,
                        frame_rate=frame_rate or 1,
                        start_time=int(start_time),
                        end_time=int(end_time) if end_time else None
                    )
                    return JSONResponse(content={
                        "status": "success",
                        "message": "Video detection completed",
                        "frames_processed": len(result) if result is not None and hasattr(result, '__len__') else 0,
                        "output_directory": "/workspace/output/",
                        "format": "masks only"
                    })
            finally:
                os.unlink(temp_video_path)
        
        # Case 3: Video file path
        elif input_path is not None:
            if not os.path.exists(input_path):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Video file not found: {input_path}"}
                )
            
            if output_format == "frames_and_masks":
                result = process_video_with_original_frames(
                    video_path=input_path,
                    output_dir='/workspace/shared/',
                    frame_rate=frame_rate or 1,
                    start_time=int(start_time),
                    end_time=int(end_time) if end_time else None
                )
                return JSONResponse(content={
                    "status": "success",
                    "message": "Video processed with frames and masks",
                    "frames_processed": result,
                    "output_directory": "/workspace/shared/",
                    "format": "original_frames + masks for LaMa"
                })
            else:
                result = process_video_direct(
                    video_path=input_path,
                    frame_rate=frame_rate or 1,
                    start_time=int(start_time),
                    end_time=int(end_time) if end_time else None
                )
                return JSONResponse(content={
                    "status": "success",
                    "message": "Video detection completed",
                    "frames_processed": len(result) if result is not None and hasattr(result, '__len__') else 0,
                    "output_directory": "/workspace/output/",
                    "format": "masks only"
                })
        
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "No input provided. Use either input_data (file) or input_path (string)"}
            )
            
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": error_message,
                "traceback": error_traceback
            }
        )

@app.post("/detect_and_inpaint")
async def detect_and_inpaint(
    video_file: UploadFile = File(..., description="Video file to process"),
    frame_rate: Optional[int] = Form(None, description="Extract every nth frame"),
    start_time: float = Form(0, description="Start time in seconds"),
    end_time: Optional[float] = Form(None, description="End time in seconds"),
    lama_service_url: str = Form("http://lama:8002", description="LaMa service URL"),
    output_dir: Optional[str] = Form(None, description="Output directory for inpainted images"),
    return_format: str = Form("files", description="Return format: 'files' or 'base64'"),
    filename_suffix: str = Form("_clean", description="Suffix for output filenames"),
    session_id: Optional[str] = Form(None, description="Session ID for tracking")
):
    """
    FRAME-BY-FRAME PIPELINE: frame → ARDCNN → LaMa → output → LOG IMMEDIATELY
    """
    try:
        import requests
        import cv2
        import numpy as np
        from io import BytesIO
        from PIL import Image
        import time
        import asyncio
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            content = await video_file.read()
            temp_video.write(content)
            temp_video_path = temp_video.name
        
        try:
            # Initialize ARDCNN model once
            from removal.video_dataloader import VideoDataLoader
            from removal.ard_cnn import ARDCNN
            from tensorflow import keras
            
            # Create model
            image_input = keras.Input(shape=(256, 512, 3), name='rain')
            ard_cnn = ARDCNN(image_input, False)
            model = keras.Model(image_input, ard_cnn.outputs)
            model.load_weights('/workspace/model/ard.40_0.00649.hdf5')
            
            # Extract and process frames one by one
            cap = cv2.VideoCapture(temp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            processed_count = 0
            all_results = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply time filtering
                current_time = frame_count / fps
                if current_time < start_time:
                    frame_count += 1
                    continue
                if end_time and current_time > end_time:
                    break
                
                # Apply frame rate filtering
                if frame_rate and frame_count % frame_rate != 0:
                    frame_count += 1
                    continue
                
                # Start total frame timing
                frame_start_time = time.time()
                
                # STEP 1: Process single frame through ARDCNN
                resized_frame = cv2.resize(frame, (512, 256))
                normalized_frame = resized_frame.astype(np.float32) / 255.0
                frame_batch = np.expand_dims(normalized_frame, axis=0)
                
                prediction = model.predict(frame_batch, verbose=0)[0]
                mask = np.where(prediction < 0.5, 0, 255).astype(np.uint8)
                
                # STEP 2: Send to LaMa
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_buffer = BytesIO()
                frame_pil.save(frame_buffer, format='PNG')
                frame_buffer.seek(0)
                
                mask_pil = Image.fromarray(mask.squeeze())
                mask_buffer = BytesIO()
                mask_pil.save(mask_buffer, format='PNG')
                mask_buffer.seek(0)
                
                files = [
                    ('images', (f'frame_{processed_count}.png', frame_buffer, 'image/png')),
                    ('masks', (f'mask_{processed_count}.png', mask_buffer, 'image/png'))
                ]
                
                lama_data = {
                    'return_format': return_format,
                    'filename_suffix': f'{filename_suffix}_frame_{processed_count}',
                    'frame_index': str(processed_count)
                }
                
                if output_dir:
                    lama_data['output_dir'] = output_dir
                
                # STEP 3: Call LaMa for this single frame
                response = requests.post(
                    f"{lama_service_url}/inpaint",
                    files=files,
                    data=lama_data,
                    timeout=30
                )
                
                # Calculate total frame latency
                frame_end_time = time.time()
                total_frame_latency = (frame_end_time - frame_start_time) * 1000
                
                # STEP 4: IMMEDIATELY send latency to main API after each frame
                def send_latency_callback():
                    try:
                        callback_data = {
                            "frame_index": processed_count,
                            "total_latency_ms": round(total_frame_latency, 2)
                        }
                        
                        # Add session_id if provided
                        if session_id:
                            callback_data["session_id"] = session_id
                        
                        callback_response = requests.post(
                            f"{HOST_CALLBACK_URL}/latency_callback",
                            json=callback_data,
                            timeout=2
                        )
                        if callback_response.status_code == 200:
                            print(f"✓ Latency logged for frame {processed_count}: {total_frame_latency:.1f}ms")
                        else:
                            print(f"⚠ Latency callback failed for frame {processed_count}: HTTP {callback_response.status_code}")
                    except Exception as callback_error:
                        print(f"⚠ Latency callback error for frame {processed_count}: {callback_error}")
                        print(f"   Attempted URL: {HOST_CALLBACK_URL}/latency_callback")
                
                # Send callback immediately (non-blocking)
                send_latency_callback()
                
                if response.status_code == 200:
                    lama_result = response.json()
                    all_results.append({
                        "frame_index": processed_count,
                        "status": "success",
                        "total_latency_ms": round(total_frame_latency, 2)
                    })
                    print(f"✅ Frame {processed_count} completed: {total_frame_latency:.1f}ms")
                else:
                    all_results.append({
                        "frame_index": processed_count,
                        "status": "failed",
                        "total_latency_ms": round(total_frame_latency, 2)
                    })
                    print(f"❌ Frame {processed_count} failed: {total_frame_latency:.1f}ms")
                
                processed_count += 1
                frame_count += 1
            
            cap.release()
            
            # Summary
            successful_frames = len([r for r in all_results if r["status"] == "success"])
            failed_frames = len([r for r in all_results if r["status"] == "failed"])
            
            return JSONResponse(content={
                "status": "success",
                "message": f"Frame-by-frame pipeline completed",
                "total_frames_processed": processed_count,
                "successful_frames": successful_frames,
                "failed_frames": failed_frames,
                "processing_mode": "frame_by_frame_realtime_logging",
                "results": all_results
            })
            
        finally:
            os.unlink(temp_video_path)
    
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": error_message,
                "traceback": error_traceback
            }
        )

@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download a processed video file"""
    file_path = f"/workspace/{filename}"
    
    print(f"Download request for: {filename}")
    print(f"Looking for file at: {file_path}")
    
    # List all files in workspace for debugging
    try:
        workspace_files = os.listdir("/workspace")
        video_files = [f for f in workspace_files if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"Available video files: {video_files}")
        
        # Also check specific directories
        if os.path.exists("/workspace/output_frames"):
            frame_files = os.listdir("/workspace/output_frames")
            print(f"Frame files in output_frames: {len(frame_files)} files")
            
    except Exception as e:
        print(f"Error listing workspace files: {e}")
    
    print(f"File exists: {os.path.exists(file_path)}")
    
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"File found! Size: {file_size} bytes")
        
        # Verify it's a valid video file
        if file_size == 0:
            print("Warning: Video file is empty!")
            return JSONResponse(
                status_code=404,
                content={"error": f"Video file is empty: {filename}"}
            )
            
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='video/mp4',
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET",
                "Access-Control-Allow-Headers": "*",
                "Cache-Control": "no-cache",
                "Content-Length": str(file_size)
            }
        )
    else:
        print(f"File not found: {file_path}")
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Video file not found: {filename}",
                "searched_path": file_path,
                "available_files": [f for f in os.listdir("/workspace") if f.endswith(('.mp4', '.avi', '.mov'))]
            }
        )


@app.get("/list_output_files")
def list_output_files():
    """List all files in the workspace directory"""
    try:
        files = []
        workspace_files = [f for f in os.listdir("/workspace") if f.endswith(('.mp4', '.avi', '.mov'))]
        for file in workspace_files:
            file_path = f"/workspace/{file}"
            file_size = os.path.getsize(file_path)
            files.append({
                "filename": file,
                "size_bytes": file_size,
                "download_url": f"/download/{file}"
            })
        
        return JSONResponse(content={
            "status": "success",
            "video_files": files,
            "total_files": len(files)
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list files: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
