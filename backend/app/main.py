from fastapi import FastAPI

app = FastAPI(
    title="Low-Light Video Processing API",
    description="API for enhancing low-light videos using CLAHE, UNet, or Selective UNet",
    version="1.0"
)

from app.routes import video_process
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174",
                   "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your video processing routes
app.include_router(video_process.router,prefix="/api")