#!/usr/bin/env python3

import os
import sys
import tempfile
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import traceback
import io
import base64
from PIL import Image
import time

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# Add the LaMa project to Python path
sys.path.append('/workspace/lama')
sys.path.append('/workspace/lama/bin')

from predict import main as predict_main
from omegaconf import OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
import torch
import yaml

app = FastAPI(title="LaMa Inpainting API")

# Configuration
LAMA_MODEL_PATH = "/workspace/lama/big-lama"
OUTPUT_DIR = "/workspace/lama/output"

# Global model cache
_model_cache = None
_train_config_cache = None

def load_model():
    """Load LaMa model once and cache it"""
    global _model_cache, _train_config_cache
    
    if _model_cache is None:
        device = torch.device("cpu")
        
        train_config_path = os.path.join(LAMA_MODEL_PATH, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        
        checkpoint_path = os.path.join(LAMA_MODEL_PATH, 'models', 'best.ckpt')
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        model.to(device)
        
        _model_cache = model
        _train_config_cache = train_config
    
    return _model_cache, _train_config_cache

def process_image_pair_direct(image_array: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
    """
    Process a single image-mask pair directly in memory using LaMa
    
    Args:
        image_array: RGB image as numpy array (H, W, 3)
        mask_array: Mask as numpy array (H, W) where white=inpaint, black=keep
        
    Returns:
        Inpainted image as numpy array (H, W, 3)
    """
    model, train_config = load_model()
    device = torch.device("cpu")
    
    # Ensure images are the same size
    if image_array.shape[:2] != mask_array.shape:
        mask_pil = Image.fromarray(mask_array)
        mask_pil = mask_pil.resize((image_array.shape[1], image_array.shape[0]))
        mask_array = np.array(mask_pil)
    
    # Ensure the image dimensions are divisible by 8 (common requirement for deep learning models)
    h, w = image_array.shape[:2]
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    
    if h != new_h or w != new_w:
        # Resize image and mask to be divisible by 8
        image_pil = Image.fromarray(image_array)
        mask_pil = Image.fromarray(mask_array)
        
        image_pil = image_pil.resize((new_w, new_h))
        mask_pil = mask_pil.resize((new_w, new_h))
        
        image_array = np.array(image_pil)
        mask_array = np.array(mask_pil)
    
    # Convert to tensor format expected by LaMa
    # Image: normalize to [0,1] and convert to CHW format
    image_tensor = torch.from_numpy(image_array.astype(np.float32) / 255.0).permute(2, 0, 1)
    
    # Mask: convert to binary and add channel dimension
    mask_tensor = torch.from_numpy((mask_array > 127).astype(np.float32))[None, ...]
    
    # Create batch
    batch = {
        'image': image_tensor[None, ...],  # Add batch dimension
        'mask': mask_tensor[None, ...]     # Add batch dimension
    }
    
    # Move to device
    batch = move_to_device(batch, device)
    
    # Run inference
    with torch.no_grad():
        batch['mask'] = (batch['mask'] > 0) * 1
        result_batch = model(batch)
        result_tensor = result_batch['inpainted'][0]  # Remove batch dimension
    
    # Convert back to numpy array
    result_array = result_tensor.permute(1, 2, 0).detach().cpu().numpy()
    result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
    
    # If we resized earlier, resize back to original dimensions
    if h != new_h or w != new_w:
        result_pil = Image.fromarray(result_array)
        result_pil = result_pil.resize((w, h))
        result_array = np.array(result_pil)
    
    return result_array

@app.get("/")
def health_check():
    return {"message": "LaMa inpainting service is running", "model_path": LAMA_MODEL_PATH}

@app.post("/inpaint")
async def inpaint(
    images: List[UploadFile] = File(..., description="Original images"),
    masks: List[UploadFile] = File(..., description="Corresponding mask images"),
    output_dir: Optional[str] = Form(None, description="Output directory (optional)"),
    return_format: str = Form("files", description="Return format: 'files' or 'base64'"),
    save_to_original_dir: bool = Form(False, description="Save to same directory as original images"),
    filename_suffix: str = Form("_inpainted", description="Suffix to add to output filenames"),
    frame_index: Optional[str] = Form(None, description="Frame index for tracking")
):
    """Unified inpainting endpoint - simplified without detailed latency tracking"""
    try:
        if len(images) != len(masks):
            return JSONResponse(
                status_code=400,
                content={"error": f"Number of images ({len(images)}) must match number of masks ({len(masks)})"}
            )
        
        # Set output directory
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model once
        load_model()
        
        results = []
        output_files = []
        
        for i, (image_file, mask_file) in enumerate(zip(images, masks)):
            # Read and process
            image_content = await image_file.read()
            mask_content = await mask_file.read()
            image_pil = Image.open(io.BytesIO(image_content)).convert('RGB')
            mask_pil = Image.open(io.BytesIO(mask_content)).convert('L')
            image_array = np.array(image_pil)
            mask_array = np.array(mask_pil)
            if image_array.shape[:2] != mask_array.shape:
                mask_pil = mask_pil.resize(image_pil.size)
                mask_array = np.array(mask_pil)
            # Measure LaMa latency
            lama_start = time.time()
            result_array = process_image_pair_direct(image_array, mask_array)
            lama_time = time.time() - lama_start
            # Save result
            original_filename = image_file.filename or f"image_{i}"
            original_name = os.path.splitext(original_filename)[0]
            output_filename = f"{original_name}{filename_suffix}.png"
            final_output_dir = output_dir
            os.makedirs(final_output_dir, exist_ok=True)
            output_path = os.path.join(final_output_dir, output_filename)
            result_pil = Image.fromarray(result_array)
            result_pil.save(output_path, format='PNG')
            if return_format == "base64":
                buffer = io.BytesIO()
                result_pil.save(buffer, format='PNG')
                result_b64 = base64.b64encode(buffer.getvalue()).decode()
                results.append({
                    "index": i,
                    "original_filename": original_filename,
                    "output_filename": output_filename,
                    "data": result_b64,
                    "format": "base64",
                    "processing_time_ms": int(lama_time * 1000)
                })
            else:
                output_files.append(output_path)
                results.append({
                    "index": i,
                    "original_filename": original_filename,
                    "output_filename": output_filename,
                    "output_path": output_path,
                    "format": "file",
                    "processing_time_ms": int(lama_time * 1000)
                })
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Processed {len(results)} image pairs successfully",
            "total_processed": len(results),
            "output_files": output_files if return_format == "files" else [],
            "results": results
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Inpainting failed: {str(e)}"}
        )

@app.post("/inpaint_video_frames")
async def inpaint_video_frames(
    video_file: UploadFile = File(..., description="Video file"),
    masks: List[UploadFile] = File(..., description="Mask files for corresponding frames"),
    frame_indices: Optional[str] = Form(None, description="Comma-separated frame indices to process"),
    return_format: str = Form("base64", description="Return format: 'base64' or 'binary'")
):
    """
    Process video frames with corresponding masks directly in memory.
    
    Args:
        video_file: Input video file
        masks: Mask files corresponding to frames
        frame_indices: Optional comma-separated indices of frames to process
        return_format: How to return results
    """
    try:
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_content = await video_file.read()
            temp_video.write(video_content)
            temp_video_path = temp_video.name
        
        try:
            # Extract frames from video
            cap = cv2.VideoCapture(temp_video_path)
            frames = []
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_idx += 1
            cap.release()
            
            # Parse frame indices if provided
            if frame_indices:
                indices = [int(x.strip()) for x in frame_indices.split(',')]
                frames = [frames[i] for i in indices if i < len(frames)]
            
            if len(frames) != len(masks):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Number of frames ({len(frames)}) must match number of masks ({len(masks)})"}
                )
            
            # Load model once
            load_model()
            
            results = []
            
            for i, (frame, mask_file) in enumerate(zip(frames, masks)):
                # Read mask
                mask_content = await mask_file.read()
                mask_pil = Image.open(io.BytesIO(mask_content)).convert('L')
                mask_array = np.array(mask_pil)
                
                # Resize mask to match frame
                if frame.shape[:2] != mask_array.shape:
                    mask_pil = mask_pil.resize((frame.shape[1], frame.shape[0]))
                    mask_array = np.array(mask_pil)
                
                # Process with LaMa
                result_array = process_image_pair_direct(frame, mask_array)
                
                if return_format == "base64":
                    result_pil = Image.fromarray(result_array)
                    buffer = io.BytesIO()
                    result_pil.save(buffer, format='PNG')
                    result_b64 = base64.b64encode(buffer.getvalue()).decode()
                    results.append({
                        "frame_index": i,
                        "filename": f"frame_{i}_inpainted.png",
                        "data": result_b64,
                        "format": "base64"
                    })
                else:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    output_path = os.path.join(OUTPUT_DIR, f"frame_{i}_inpainted.png")
                    cv2.imwrite(output_path, cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR))
                    results.append({
                        "frame_index": i,
                        "filename": f"frame_{i}_inpainted.png",
                        "download_url": f"/download/frame_{i}_inpainted.png",
                        "format": "file"
                    })
            
            return JSONResponse(content={
                "status": "success",
                "message": f"Processed {len(results)} video frames successfully",
                "results": results,
                "processing_mode": "direct_memory"
            })
            
        finally:
            os.unlink(temp_video_path)
            
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Video frame inpainting failed: {error_message}",
                "traceback": error_traceback
            }
        )

@app.post("/inpaint_files")
async def inpaint_files(
    image_paths: List[str] = Form(..., description="Paths to original image files"),
    mask_paths: List[str] = Form(..., description="Paths to corresponding mask files"),
    output_dir: Optional[str] = Form(None, description="Output directory (optional)"),
    save_to_original_dir: bool = Form(True, description="Save to same directory as original images"),
    filename_suffix: str = Form("_inpainted", description="Suffix to add to output filenames")
):
    """
    Inpainting endpoint that works with file paths and can save outputs
    in the same directory as the original images with normal PNG format.
    
    Args:
        image_paths: List of paths to original image files
        mask_paths: List of paths to corresponding mask files 
        output_dir: Directory to save output files (used if save_to_original_dir=False)
        save_to_original_dir: If True, save outputs in same directory as original images
        filename_suffix: Suffix to add to output filenames (default: '_inpainted')
    """
    try:
        if len(image_paths) != len(mask_paths):
            return JSONResponse(
                status_code=400,
                content={"error": f"Number of image paths ({len(image_paths)}) must match number of mask paths ({len(mask_paths)})"}
            )
        
        # Load model once
        load_model()
        
        results = []
        output_files = []
        
        for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            # Verify files exist
            if not os.path.exists(image_path):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Image file not found: {image_path}"}
                )
            
            if not os.path.exists(mask_path):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Mask file not found: {mask_path}"}
                )
            
            # Load images
            image_pil = Image.open(image_path).convert('RGB')
            mask_pil = Image.open(mask_path).convert('L')
            
            image_array = np.array(image_pil)
            mask_array = np.array(mask_pil)
            
            # Ensure same size
            if image_array.shape[:2] != mask_array.shape:
                mask_pil = mask_pil.resize(image_pil.size)
                mask_array = np.array(mask_pil)
            
            # Process with LaMa
            result_array = process_image_pair_direct(image_array, mask_array)
            
            # Generate output path
            image_dir = os.path.dirname(image_path)
            image_basename = os.path.basename(image_path)
            image_name = os.path.splitext(image_basename)[0]
            output_filename = f"{image_name}{filename_suffix}.png"
            
            if save_to_original_dir:
                final_output_dir = image_dir
            else:
                final_output_dir = output_dir or OUTPUT_DIR
            
            os.makedirs(final_output_dir, exist_ok=True)
            output_path = os.path.join(final_output_dir, output_filename)
            
            # Save as normal PNG file
            result_pil = Image.fromarray(result_array)
            result_pil.save(output_path, format='PNG')
            
            output_files.append(output_path)
            results.append({
                "index": i,
                "original_image_path": image_path,
                "original_mask_path": mask_path,
                "output_filename": output_filename,
                "output_path": output_path,
                "output_directory": final_output_dir
            })
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Processed {len(results)} image pairs successfully",
            "output_files": output_files,
            "results": results,
            "processing_mode": "file_based",
            "filename_suffix": filename_suffix,
            "save_to_original_dir": save_to_original_dir,
            "total_processed": len(results)
        })
        
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": f"File-based inpainting failed: {error_message}",
                "traceback": error_traceback
            }
        )

@app.get("/download/{filename}")
async def download_result(filename: str):
    """Download a result file."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"File not found: {filename}"}
        )

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8002)
