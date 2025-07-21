# Backend Integration Guide

## MongoDB API Endpoints

Add these endpoints to your existing FastAPI backend to enable analytics functionality.

### 1. Update your main FastAPI file

Add these imports to your main FastAPI file (where you have `app = FastAPI()`):

```python
from backend_integration.video_process_routes import router
```

### 2. Include the router

```python
app.include_router(router, prefix="/api")
```

### 3. File Structure

Place the `video_process_routes.py` file in your backend directory and import it.

## Endpoints Provided

### GET /api/logs
- **Purpose**: Fetch all processing logs from MongoDB
- **Returns**: Array of log objects (newest first, limited to 50)
- **Frontend Usage**: Powers the analytics dashboard and processing history

### GET /api/logs/latest  
- **Purpose**: Get the most recent processing session
- **Returns**: Single log object or null
- **Frontend Usage**: Shows current session details in analytics

### GET /api/logs/method/{method_name}
- **Purpose**: Filter logs by processing method
- **Returns**: Array of logs for specific method
- **Frontend Usage**: Method-specific analytics

### GET /api/stats
- **Purpose**: Aggregate processing statistics
- **Returns**: Summary stats (total processed, success rate, avg times)
- **Frontend Usage**: Dashboard overview metrics

## Database Schema

The endpoints expect your existing MongoDB structure from `db_logger.py`:

```python
{
    "timestamp": datetime,
    "process_category": str,
    "model_used": str,
    "video": {
        "input_file": str,
        "output_file": str,
        "duration": float,
        "resolution": str,
        "fps": float,
        "total_frames": int,
        "processed_frames": int
    },
    "performance": {
        "total_time": float,
        "avg_delay_per_frame": float,
        "cpu_usage_percent": float,
        "gpu_usage_percent": float,
        "device_used": str,
        "device_specs": {
            "cpu": str,
            "gpu": str
        }
    },
    "status": str,
    "error_message": str
}
```

## Integration Steps

1. Copy `video_process_routes.py` to your backend directory
2. Import and include the router in your main FastAPI app
3. Ensure MongoDB is running on `mongodb://localhost:27017`
4. Your existing `db_logger.py` will continue to work unchanged
5. Frontend will automatically start showing real-time analytics data

## CORS Configuration

Make sure your CORS middleware allows the analytics endpoints:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

The frontend is now ready to display real-time MongoDB data!