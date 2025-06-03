"""
API routes for the face liveness detection system.
"""

import io
import base64
from typing import Optional, List

import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.detector import LivenessDetector
from utils.logging import setup_logger

logger = setup_logger("liveness_api_routes")

# Pydantic models for request/response
class LivenessResponse(BaseModel):
    is_live: bool
    live_probability: float
    spoof_probability: float
    message: str
    status: str  # Added status field for machine-readable output

class Base64ImageRequest(BaseModel):
    image: str
    threshold: Optional[float] = 0.5

# Initialize router
router = APIRouter(prefix="/detect", tags=["detection"])

# Initialize detector
detector = None

def get_detector():
    """Get or initialize the liveness detector."""
    global detector
    if detector is None:
        logger.info("Initializing liveness detector on first request...")
        detector = LivenessDetector()
        logger.info("Liveness detector initialized")
    return detector

def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64 image string to a numpy array.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        np.ndarray: Decoded image as RGB numpy array
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
            
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to numpy array using OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return rgb_img
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise ValueError(f"Invalid base64 image: {e}")

@router.post("/image", response_model=LivenessResponse)
async def detect_from_image(file: UploadFile = File(...), threshold: float = Form(0.5)):
    """
    Detect liveness from an uploaded image file.
    
    Args:
        file: Uploaded image file
        threshold: Probability threshold to classify as live (0.0-1.0)
        
    Returns:
        LivenessResponse: Detection results
    """
    try:
        # Get the detector
        liveness_detector = get_detector()
        
        # Read and validate the file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read the image data
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect liveness
        probs = liveness_detector.detect(rgb_img)[0]  # [p_spoof, p_live]
        is_live = probs[1] > threshold
        
        # Prepare response with improved message format
        status = "LIVE" if is_live else "SPOOF"
        message = "Real_face_detected" if is_live else "Fake_face_detected"
        
        result = LivenessResponse(
            is_live=bool(is_live),
            live_probability=float(probs[1]),
            spoof_probability=float(probs[0]),
            message=message,
            status=status
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/base64", response_model=LivenessResponse)
async def detect_from_base64(request: Base64ImageRequest):
    """
    Detect liveness from a base64 encoded image.
    
    Args:
        request: Request containing base64 image and optional threshold
        
    Returns:
        LivenessResponse: Detection results
    """
    try:
        # Get the detector
        liveness_detector = get_detector()
        
        # Decode the base64 image
        try:
            rgb_img = decode_base64_image(request.image)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if rgb_img is None or rgb_img.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Detect liveness
        probs = liveness_detector.detect(rgb_img)[0]  # [p_spoof, p_live]
        is_live = probs[1] > request.threshold
        
        # Prepare response with improved message format
        status = "LIVE" if is_live else "SPOOF"
        message = "Real_face_detected" if is_live else "Fake_face_detected"
        
        result = LivenessResponse(
            is_live=bool(is_live),
            live_probability=float(probs[1]),
            spoof_probability=float(probs[0]),
            message=message,
            status=status
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing base64 image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/batch", response_model=List[LivenessResponse])
async def detect_batch(
    files: List[UploadFile] = File(...), 
    threshold: float = Form(0.5),
    background_tasks: BackgroundTasks = None
):
    """
    Batch process multiple images for liveness detection.
    
    Args:
        files: List of uploaded image files
        threshold: Probability threshold to classify as live (0.0-1.0)
        
    Returns:
        List[LivenessResponse]: List of detection results for each image
    """
    try:
        # Get the detector
        liveness_detector = get_detector()
        
        # Process each image
        results = []
        for file in files:
            # Check if the file is an image
            if not file.content_type.startswith("image/"):
                results.append(LivenessResponse(
                    is_live=False,
                    live_probability=0.0,
                    spoof_probability=1.0,
                    message=f"Error: File {file.filename} is not an image",
                    status="ERROR"
                ))
                continue
                
            try:
                # Read the image data
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    results.append(LivenessResponse(
                        is_live=False,
                        live_probability=0.0,
                        spoof_probability=1.0,
                        message=f"Error: Invalid image file {file.filename}",
                        status="ERROR"
                    ))
                    continue
                
                # Convert BGR to RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detect liveness
                probs = liveness_detector.detect(rgb_img)[0]
                is_live = probs[1] > threshold
                
                # Prepare response with improved message format
                status = "LIVE" if is_live else "SPOOF"
                message = "Real_face_detected" if is_live else "Fake_face_detected"
                
                # Add result
                results.append(LivenessResponse(
                    is_live=bool(is_live),
                    live_probability=float(probs[1]),
                    spoof_probability=float(probs[0]),
                    message=message,
                    status=status
                ))
                
            except Exception as e:
                logger.error(f"Error processing image {file.filename}: {e}")
                results.append(LivenessResponse(
                    is_live=False,
                    live_probability=0.0,
                    spoof_probability=1.0,
                    message=f"Error processing image {file.filename}: {str(e)}",
                    status="ERROR"
                ))
        
        # Clean up
        if background_tasks:
            background_tasks.add_task(lambda: [f.file.close() for f in files if hasattr(f, 'file')])
            
        return results
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error in batch processing: {str(e)}") 