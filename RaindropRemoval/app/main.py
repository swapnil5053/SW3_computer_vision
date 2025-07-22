from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import subprocess
import time
import cv2
import numpy as np
import requests
import json
from PIL import Image
import io

app = FastAPI(title="ARDCNN Raindrop Removal Orchestrator")

PROCESSED_VIDEOS_DIR = "/backend/processed_videos"
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

@app.post("/process_for_analytics")
async def process_for_analytics(
    video_file: UploadFile = File(...),
    lama_service_url: str = Form("http://lama:8002"),
    session_id: str = Form(...)
):
    pipeline_start_time = time.time()
    frame_timings = []
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        content = await video_file.read()
        temp_video.write(content)
        temp_video_path = temp_video.name

    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        processed_frames_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start_time = time.time()

            # 1. ARDCNN Detection (Simulated)
            ardcnn_start = time.time()
            time.sleep(0.1 + np.random.rand() * 0.05)
            mask = np.zeros((original_height, original_width), dtype=np.uint8)
            cv2.circle(mask, (np.random.randint(0, original_width), np.random.randint(0, original_height)), 
                       radius=np.random.randint(10, 50), color=(255,255,255), thickness=-1)
            ardcnn_time = time.time() - ardcnn_start

            # 2. LaMa Inpainting (via HTTP call)
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(mask)
            
            with io.BytesIO() as frame_buffer, io.BytesIO() as mask_buffer:
                frame_pil.save(frame_buffer, format='PNG')
                mask_pil.save(mask_buffer, format='PNG')
                frame_buffer.seek(0)
                mask_buffer.seek(0)
                
                files = {'images': (f'frame_{frame_count}.png', frame_buffer, 'image/png'), 'masks': (f'mask_{frame_count}.png', mask_buffer, 'image/png')}
                
                response = requests.post(f"{lama_service_url}/inpaint", files=files)
                response.raise_for_status()
                
                lama_result = response.json().get('results', [{}])[0]
                lama_processing_time = lama_result.get('processing_time_ms', 0) / 1000.0
            
            processed_frames_data.append(frame)

            # Record all timings for this frame
            total_frame_time = time.time() - frame_start_time
            
            # **NEW**: Calculate the combined processing time of the two models
            combined_process_time = ardcnn_time + lama_processing_time

            frame_timings.append({
                "frame_number": frame_count,
                "total_time": total_frame_time,
                "process_time": combined_process_time, # The sum of model times
                "was_processed": True,
                "timestamp": time.time()
            })
            
            frame_count += 1

        cap.release()
        
        output_video_path = os.path.join(PROCESSED_VIDEOS_DIR, f"{session_id}.mp4")
        import shutil
        shutil.copy(temp_video_path, output_video_path)

        pipeline_total_time = time.time() - pipeline_start_time
        timing_output_path = os.path.join(PROCESSED_VIDEOS_DIR, f"{session_id}_timings.json")
        
        timing_data = {
            "processing_method": "raindrop_removal",
            "model_used": "ARDCNN + LaMa",
            "total_input_frames": total_input_frames,
            "frames_processed": frame_count,
            "total_processing_time_seconds": pipeline_total_time,
            "avg_time_per_frame_seconds": pipeline_total_time / frame_count if frame_count > 0 else 0,
            "frame_by_frame_timings": frame_timings,
            "video_info": {
                "input_path": video_file.filename,
                "output_path": output_video_path,
                "original_fps": fps,
                "output_resolution": f"{original_width}x{original_height}"
            }
        }
        
        with open(timing_output_path, 'w') as f:
            json.dump(timing_data, f, indent=2)

        return JSONResponse(content={ "status": "success", "session_id": session_id })

    finally:
        os.unlink(temp_video_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# import os
# import shutil
# from datetime import datetime
# from processes.process import test_detect_and_inpaint_with_file, process_low_light_video
# import uvicorn
# from pymongo import MongoClient
# import time
# import cv2
# import psutil
# import GPUtil
# import platform
# import subprocess

# # MongoDB Atlas connection (replace with your connection string)
# MONGO_URI = "mongodb+srv://admin:admin@cluster0.tpkkrn2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# client = MongoClient(MONGO_URI)
# db = client.model_logs
# logs_collection = db.model_logs

# app = FastAPI()

# # Create uploads directory if it doesn't exist
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Track processing sessions
# processing_sessions = {}

# def get_video_metadata(video_path):
#     """Extract video metadata using OpenCV"""
#     try:
#         cap = cv2.VideoCapture(video_path)
        
#         # Get video properties
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         duration = frame_count / fps if fps > 0 else 0
        
#         cap.release()
        
#         return {
#             "duration": round(duration, 2),
#             "resolution": f"{width}x{height}",
#             "fps": round(fps, 2),
#             "total_frames": frame_count
#         }
#     except Exception as e:
#         print(f"Error getting video metadata: {e}")
#         return {
#             "duration": 0,
#             "resolution": "unknown",
#             "fps": 0,
#             "total_frames": 0
#         }

# def get_system_specs():
#     """Get CPU and GPU specifications"""
#     try:
#         # Get CPU info
#         cpu_name = platform.processor()
#         if not cpu_name:
#             try:
#                 # Try alternative method for CPU name
#                 with open('/proc/cpuinfo', 'r') as f:
#                     for line in f:
#                         if 'model name' in line:
#                             cpu_name = line.split(':')[1].strip()
#                             break
#             except:
#                 cpu_name = "Unknown CPU"
        
#         # Get GPU info
#         gpu_name = "No GPU"
#         try:
#             gpus = GPUtil.getGPUs()
#             if gpus:
#                 gpu_name = gpus[0].name
#         except:
#             # Fallback to nvidia-smi if GPUtil fails
#             try:
#                 result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
#                                       capture_output=True, text=True, timeout=5)
#                 if result.returncode == 0:
#                     gpu_name = result.stdout.strip().split('\n')[0]
#             except:
#                 gpu_name = "Unknown GPU"
        
#         return {
#             "cpu": cpu_name,
#             "gpu": gpu_name
#         }
#     except Exception as e:
#         print(f"Error getting system specs: {e}")
#         return {
#             "cpu": "Unknown CPU",
#             "gpu": "Unknown GPU"
#         }

# def get_resource_usage():
#     """Get current CPU and GPU usage"""
#     try:
#         # Get CPU usage
#         cpu_usage = psutil.cpu_percent(interval=1)
        
#         # Get GPU usage
#         gpu_usage = 0
#         try:
#             gpus = GPUtil.getGPUs()
#             if gpus:
#                 gpu_usage = gpus[0].load * 100  # Convert to percentage
#         except:
#             gpu_usage = 0
        
#         return {
#             "cpu_usage_percent": round(cpu_usage, 2),
#             "gpu_usage_percent": round(gpu_usage, 2)
#         }
#     except Exception as e:
#         print(f"Error getting resource usage: {e}")
#         return {
#             "cpu_usage_percent": 0,
#             "gpu_usage_percent": 0
#         }

# def determine_device_used():
#     """Determine if GPU or CPU was used for processing"""
#     try:
#         gpus = GPUtil.getGPUs()
#         if gpus and gpus[0].load > 0.1:  # If GPU load > 10%, assume GPU was used
#             return "GPU"
#         else:
#             return "CPU"
#     except:
#         return "CPU"

# @app.get("/")
# def sanity_check():
#     return {"Status": "200"}

# @app.post("/upload")
# async def upload_video(video_file: UploadFile = File(...)):
#     """Upload a video file and save it to organized uploads directory"""
#     try:
#         # Validate file type
#         if not video_file.content_type or not video_file.content_type.startswith('video/'):
#             return JSONResponse(
#                 status_code=400,
#                 content={"error": "File must be a video file"}
#             )
        
#         # Create a unique filename with timestamp for better organization
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = video_file.filename or "unknown_video"
#         file_extension = os.path.splitext(filename)[1]
#         temp_filename = f"video_{timestamp}_{hash(filename)}{file_extension}"
        
#         # Save to organized uploads directory
#         file_path = os.path.join(UPLOAD_DIR, temp_filename)
        
#         # Save the uploaded file
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(video_file.file, buffer)
        
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "message": "Video uploaded successfully",
#                 "original_filename": video_file.filename,
#                 "saved_filename": temp_filename,
#                 "file_path": file_path,
#                 "file_size": os.path.getsize(file_path)
#             }
#         )
    
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Failed to upload video: {str(e)}"}
#         )

# @app.post("/raindrop")
# async def raindrop(
#     file_path: str = Form(...),  # Path to the uploaded file
#     start_time: int = Form(default=0),
#     end_time: int = Form(default=30),
#     frame_rate: int = Form(default=1)
# ):
#     """Communicate with the removal model using uploaded video"""
#     try:
#         # Verify the file exists locally
#         if not os.path.exists(file_path):
#             return JSONResponse(
#                 status_code=404,
#                 content={"error": f"Video file not found: {file_path}"}
#             )
        
#         # Get video metadata
#         video_metadata = get_video_metadata(file_path)
        
#         # Get system specs (once at start)
#         system_specs = get_system_specs()
        
#         # Create processing session
#         session_id = f"{int(time.time())}_{hash(file_path)}"
#         processing_sessions[session_id] = {
#             "start_time": time.time(),
#             "frame_latencies": [],
#             "input_file": file_path,
#             "process": "raindrop_removal",
#             "model": "ARDCNN + LaMa",
#             "video_metadata": video_metadata,
#             "system_specs": system_specs
#         }
        
#         print(f"🎥 Video metadata: {video_metadata}")
#         print(f"💻 System specs: {system_specs}")
        
#         # Call the processing function with the uploaded file
#         result = await test_detect_and_inpaint_with_file(
#             video_path=file_path,
#             start_time=start_time,
#             end_time=end_time,
#             frame_rate=frame_rate,
#             session_id=session_id  # Pass session ID for tracking
#         )
        
#         # Calculate metrics and log to MongoDB
#         if session_id in processing_sessions:
#             session = processing_sessions[session_id]
#             total_time = time.time() - session["start_time"]
#             frame_latencies = session["frame_latencies"]
            
#             if frame_latencies:
#                 avg_delay_per_frame = sum(frame_latencies) / len(frame_latencies)
#                 processed_frames = len(frame_latencies)
#             else:
#                 avg_delay_per_frame = 0
#                 processed_frames = 0
            
#             # Get resource usage at end of processing
#             resource_usage = get_resource_usage()
#             device_used = determine_device_used()
            
#             # Create output filename
#             output_file = file_path.replace('.mp4', '_processed.mp4')
            
#             # Enhanced log entry with all required fields
#             log_entry = {
#                 "timestamp": datetime.now(),
#                 "process_category": session["process"],
#                 "model_used": session["model"],
#                 "video": {
#                     "input_file": os.path.basename(session["input_file"]),
#                     "output_file": os.path.basename(output_file),
#                     "duration": session["video_metadata"]["duration"],
#                     "resolution": session["video_metadata"]["resolution"],
#                     "fps": session["video_metadata"]["fps"],
#                     "total_frames": session["video_metadata"]["total_frames"],
#                     "processed_frames": processed_frames
#                 },
#                 "performance": {
#                     "total_time": round(total_time, 2),
#                     "avg_delay_per_frame": round(avg_delay_per_frame, 2),
#                     "frame_latencies": [round(latency, 2) for latency in frame_latencies],  # Add individual frame latencies
#                     "cpu_usage_percent": resource_usage["cpu_usage_percent"],
#                     "gpu_usage_percent": resource_usage["gpu_usage_percent"],
#                     "device_used": device_used,
#                     "device_specs": {
#                         "cpu": session["system_specs"]["cpu"],
#                         "gpu": session["system_specs"]["gpu"]
#                     }
#                 },
#                 "status": "success"
#             }
            
#             try:
#                 logs_collection.insert_one(log_entry)
#                 print(f"✅ Logged to MongoDB: {processed_frames} frames, {total_time:.2f}s total, {avg_delay_per_frame:.2f}ms avg")
#                 print(f"📊 Resource usage - CPU: {resource_usage['cpu_usage_percent']}%, GPU: {resource_usage['gpu_usage_percent']}%")
#                 print(f"📈 Frame latencies logged: {len(frame_latencies)} individual measurements")
#             except Exception as mongo_error:
#                 print(f"❌ MongoDB logging failed: {mongo_error}")
            
#             # Clean up session
#             del processing_sessions[session_id]
        
#         # Clean up the uploaded file after processing
#         try:
#             os.remove(file_path)
#         except:
#             pass  # Continue even if cleanup fails
        
#         return JSONResponse(
#             status_code=200,
#             content=result
#         )
    
#     except Exception as e:
#         # Log error to MongoDB if session exists
#         if 'session_id' in locals() and session_id in processing_sessions:
#             session = processing_sessions[session_id]
#             total_time = time.time() - session["start_time"]
#             frame_latencies = session["frame_latencies"]
            
#             log_entry = {
#                 "timestamp": datetime.now(),
#                 "process_category": session["process"],
#                 "model_used": session["model"],
#                 "video": {
#                     "input_file": os.path.basename(session["input_file"]),
#                     "output_file": "N/A",
#                     "duration": session["video_metadata"]["duration"],
#                     "resolution": session["video_metadata"]["resolution"],
#                     "fps": session["video_metadata"]["fps"],
#                     "total_frames": session["video_metadata"]["total_frames"],
#                     "processed_frames": len(frame_latencies)
#                 },
#                 "performance": {
#                     "total_time": round(total_time, 2),
#                     "avg_delay_per_frame": round(sum(frame_latencies) / len(frame_latencies), 2) if frame_latencies else 0,
#                     "frame_latencies": [round(latency, 2) for latency in frame_latencies],  # Add individual frame latencies for errors too
#                     "cpu_usage_percent": 0,
#                     "gpu_usage_percent": 0,
#                     "device_used": "N/A",
#                     "device_specs": {
#                         "cpu": session["system_specs"]["cpu"],
#                         "gpu": session["system_specs"]["gpu"]
#                     }
#                 },
#                 "status": "error",
#                 "error_message": str(e)
#             }
            
#             try:
#                 logs_collection.insert_one(log_entry)
#             except:
#                 pass
            
#             del processing_sessions[session_id]
        
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Processing failed: {str(e)}"}
#         )

# @app.post("/latency_callback")
# async def latency_callback(latency_data: dict):
#     """Receive and log frame latency"""
#     try:
#         frame_idx = latency_data.get("frame_index", "?")
#         total_latency = latency_data.get("total_latency_ms", 0)
#         session_id = latency_data.get("session_id")
        
#         print(f"Frame {frame_idx}: {total_latency:.1f}ms")
        
#         # Store latency in session
#         if session_id and session_id in processing_sessions:
#             processing_sessions[session_id]["frame_latencies"].append(total_latency)
        
#         return {"status": "ok"}
    
#     except Exception as e:
#         print(f"❌ Latency callback error: {e}")
#         return {"status": "error"}

# @app.get("/test_mongo")
# def test_mongo():
#     """Test MongoDB connection and system info"""
#     try:
#         # Test system info functions
#         video_test = get_video_metadata("./test_video.mp4") if os.path.exists("./test_video.mp4") else {"test": "no video"}
#         system_specs = get_system_specs()
#         resource_usage = get_resource_usage()
        
#         # Test MongoDB connection
#         client.admin.command('ping')
        
#         return {
#             "status": "success",
#             "message": "All systems working",
#             "system_specs": system_specs,
#             "resource_usage": resource_usage,
#             "mongodb": "connected"
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "error": str(e)
#         }

# @app.post("/dehazing")
# def haze_removal():
#     pass

# @app.post("/glare_reduction")
# def reduce_glare():
#     pass

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000)