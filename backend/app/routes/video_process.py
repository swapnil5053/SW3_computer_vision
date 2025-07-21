from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from app.services.processor import process_video_file
from pymongo import MongoClient
from datetime import datetime
import os
import json
import subprocess

# Create a router object
router = APIRouter()

# MongoDB connection (same as your db_logger.py)
client = MongoClient("mongodb://localhost:27017")
db = client["video_processing_logs"]
logs_collection = db["processing_logs"]

@router.post("/process_video/")
async def process_video(
    file: UploadFile = File(...),
    method: str = Form(...)
):
    """
    Accepts a video file and a processing method.
    Runs the appropriate ML script and returns the output video.
    """

    # Save the uploaded video temporarily
    input_path = f"uploaded_videos/{file.filename}"
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    # Call processing logic
    output_path = process_video_file(input_path, method)

    # Check if processing was successful
    if os.path.exists(output_path):
        return FileResponse(
            output_path,
            media_type='video/mp4',
            filename=os.path.basename(output_path)
        )
    else:
        return {"status": "error", "message": "Video processing failed or invalid method selected."}

@router.get("/frame_timings/{session_id}")
async def get_frame_timings(session_id: str):
    """
    Get frame-by-frame timing data for a specific processing session.
    The session_id should be the output filename with .mp4 extension.
    """
    try:
        print(f"🔍 Looking for timing data for session_id: {session_id}")
        
        # Look for timing file based on session_id (which includes .mp4)
        timing_file_path = f"processed_videos/{session_id}_timings.json"
        print(f"Checking timing file path: {timing_file_path}")
        
        if not os.path.exists(timing_file_path):
            print(f"Timing file not found: {timing_file_path}")
            
            # Try alternative naming patterns
            alternative_paths = [
                f"processed_videos/{session_id.replace('.mp4', '')}_timings.json",
                f"{session_id}_timings.json",
                f"{session_id.replace('.mp4', '')}_timings.json"
            ]
            
            for alt_path in alternative_paths:
                print(f"Trying alternative path: {alt_path}")
                if os.path.exists(alt_path):
                    timing_file_path = alt_path
                    print(f"Found timing file at: {alt_path}")
                    break
            else:
                # List available files for debugging
                try:
                    available_files = os.listdir("processed_videos/")
                    timing_files = [f for f in available_files if f.endswith("_timings.json")]
                    print(f"Available timing files: {timing_files}")
                except:
                    print("Could not list processed_videos directory")
                
                return {"error": f"Timing data not found for session: {session_id}"}
        
        print(f"📖 Reading timing file: {timing_file_path}")
        with open(timing_file_path, 'r') as f:
            timing_data = json.load(f)
        
        print(f"Successfully loaded timing data with {len(timing_data.get('frame_by_frame_timings', []))} frames")
        return timing_data
    except Exception as e:
        print(f" Error fetching frame timings: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to fetch frame timing data: {str(e)}"}
@router.get("/logs")
async def get_processing_logs():
    """
    Fetch all processing logs from MongoDB.
    Returns logs sorted by timestamp (newest first).
    """
    try:
        # Query MongoDB for all logs, sorted by timestamp (newest first)
        logs = list(logs_collection.find(
            {},  # No filter - get all logs
            {"_id": 0}  # Exclude MongoDB's _id field from response
        ).sort("timestamp", -1).limit(50))  # Limit to 50 most recent logs
        
        # Convert datetime objects to ISO strings for JSON serialization
        for log in logs:
            if isinstance(log.get("timestamp"), datetime):
                log["timestamp"] = log["timestamp"].isoformat()
        
        return logs
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return {"error": "Failed to fetch processing logs"}

@router.get("/logs/latest")
async def get_latest_log():
    """
    Fetch the most recent processing log from MongoDB.
    Returns the latest log entry or null if no logs exist.
    """
    try:
        # Query MongoDB for the most recent log
        latest_log = logs_collection.find_one(
            {},  # No filter
            {"_id": 0}  # Exclude MongoDB's _id field
        , sort=[("timestamp", -1)])  # Sort by timestamp descending
        
        if latest_log:
            # Convert datetime to ISO string for JSON serialization
            if isinstance(latest_log.get("timestamp"), datetime):
                latest_log["timestamp"] = latest_log["timestamp"].isoformat()
            return latest_log
        else:
            return None
    except Exception as e:
        print(f"Error fetching latest log: {e}")
        return {"error": "Failed to fetch latest processing log"}

@router.get("/logs/method/{method_name}")
async def get_logs_by_method(method_name: str):
    """
    Fetch processing logs filtered by method.
    Returns logs for a specific processing method.
    """
    try:
        logs = list(logs_collection.find(
            {"process_category": method_name},  # Filter by method
            {"_id": 0}
        ).sort("timestamp", -1).limit(20))
        
        # Convert datetime objects to ISO strings
        for log in logs:
            if isinstance(log.get("timestamp"), datetime):
                log["timestamp"] = log["timestamp"].isoformat()
        
        return logs
    except Exception as e:
        print(f"Error fetching logs by method: {e}")
        return {"error": f"Failed to fetch logs for method: {method_name}"}

@router.get("/stats")
async def get_processing_stats():
    """
    Get aggregate statistics from processing logs.
    Returns summary statistics about processing performance.
    """
    try:
        # Get total number of processed videos
        total_processed = logs_collection.count_documents({})
        
        # Get success rate
        successful = logs_collection.count_documents({"status": "success"})
        success_rate = (successful / total_processed * 100) if total_processed > 0 else 0
        
        # Get average processing time
        pipeline = [
            {"$match": {"status": "success"}},
            {"$group": {
                "_id": None,
                "avg_processing_time": {"$avg": "$performance.total_time"},
                "avg_frames_per_second": {"$avg": {"$divide": ["$video.total_frames", "$performance.total_time"]}}
            }}
        ]
        
        avg_stats = list(logs_collection.aggregate(pipeline))
        avg_processing_time = avg_stats[0]["avg_processing_time"] if avg_stats else 0
        avg_fps = avg_stats[0]["avg_frames_per_second"] if avg_stats else 0
        
        # Get method distribution
        method_pipeline = [
            {"$group": {
                "_id": "$process_category",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]
        
        method_distribution = list(logs_collection.aggregate(method_pipeline))
        
        return {
            "total_processed": total_processed,
            "successful_processed": successful,
            "success_rate": round(success_rate, 2),
            "average_processing_time": round(avg_processing_time, 2) if avg_processing_time else 0,
            "average_fps": round(avg_fps, 2) if avg_fps else 0,
            "method_distribution": method_distribution
        }
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return {"error": "Failed to fetch processing statistics"}