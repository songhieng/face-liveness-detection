"""
Integration module for the Face Liveness Detection System.

This module provides simple functions to use the liveness detection system
directly in other projects without needing to use the API.
"""

import os
import sys
import numpy as np
import cv2
from typing import Dict, Union, List, Tuple, Optional
from PIL import Image
import base64
import time
from datetime import datetime
from pathlib import Path

from src.core.detector import LivenessDetector
from utils.logging import setup_logger
from config.settings import PROJECT_ROOT

logger = setup_logger("liveness_integration")

# Singleton detector instance
_detector = None

# Ensure output directory exists
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "webcam_frames")

def get_detector(model_path: Optional[str] = None, device: Optional[str] = None) -> LivenessDetector:
    """
    Get or initialize the liveness detector singleton.
    
    Args:
        model_path: Optional path to model checkpoint
        device: Optional device to use (cuda or cpu)
        
    Returns:
        LivenessDetector: The detector instance
    """
    global _detector
    if _detector is None:
        logger.info("Initializing liveness detector...")
        _detector = LivenessDetector(model_path=model_path, device=device)
        logger.info("Liveness detector initialized")
    return _detector

def detect_from_image(image: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Detect liveness from a numpy image array.
    
    Args:
        image: RGB image as numpy array (HxWxC)
        threshold: Threshold for liveness detection (default: 0.5)
        
    Returns:
        Dict: Detection result with keys:
            - is_live: bool
            - live_probability: float
            - spoof_probability: float
            - message: str
            - status: str ("LIVE", "SPOOF")
    """
    detector = get_detector()
    
    # Ensure image is RGB
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB format with shape (H, W, 3)")
    
    # If image is BGR (from OpenCV), convert to RGB
    if image.dtype == np.uint8 and image.shape[2] == 3:
        # This is safe since we already know it's (H, W, 3)
        # If it came from OpenCV's imread, it's BGR format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Run detection
    probs = detector.detect(image_rgb)[0]  # [p_spoof, p_live]
    is_live = probs[1] > threshold
    
    # Create result dictionary
    status = "LIVE" if is_live else "SPOOF"
    message = "Real_face_detected" if is_live else "Fake_face_detected"
    
    return {
        "is_live": bool(is_live),
        "live_probability": float(probs[1]),
        "spoof_probability": float(probs[0]),
        "message": message,
        "status": status
    }

def detect_from_file(file_path: str, threshold: float = 0.5) -> Dict:
    """
    Detect liveness from an image file.
    
    Args:
        file_path: Path to the image file
        threshold: Threshold for liveness detection (default: 0.5)
        
    Returns:
        Dict: Detection result (same format as detect_from_image)
    """
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image from {file_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        return detect_from_image(image_rgb, threshold)
    
    except Exception as e:
        logger.error(f"Error processing image file {file_path}: {e}")
        return {
            "is_live": False,
            "live_probability": 0.0,
            "spoof_probability": 1.0,
            "message": f"Error: {str(e)}",
            "status": "ERROR"
        }

def detect_from_base64(base64_string: str, threshold: float = 0.5) -> Dict:
    """
    Detect liveness from a base64 encoded image.
    
    Args:
        base64_string: Base64 encoded image string
        threshold: Threshold for liveness detection (default: 0.5)
        
    Returns:
        Dict: Detection result (same format as detect_from_image)
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
        
        if img is None:
            raise ValueError("Could not decode base64 image")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run detection
        return detect_from_image(image_rgb, threshold)
    
    except Exception as e:
        logger.error(f"Error processing base64 image: {e}")
        return {
            "is_live": False,
            "live_probability": 0.0,
            "spoof_probability": 1.0,
            "message": f"Error: {str(e)}",
            "status": "ERROR"
        }

def detect_from_batch(file_paths: List[str], threshold: float = 0.5) -> List[Dict]:
    """
    Batch process multiple image files.
    
    Args:
        file_paths: List of paths to image files
        threshold: Threshold for liveness detection (default: 0.5)
        
    Returns:
        List[Dict]: List of detection results
    """
    return [detect_from_file(file_path, threshold) for file_path in file_paths]

def detect_from_webcam(camera_id: int = 0, threshold: float = 0.5, display: bool = True, save_frames: bool = True) -> None:
    """
    Run liveness detection on webcam feed with display and optional frame saving.
    
    Args:
        camera_id: Camera device ID (default: 0)
        threshold: Threshold for liveness detection (default: 0.5)
        display: Whether to display the webcam feed (default: True)
        save_frames: Whether to save frames with results (default: True)
        
    Returns:
        None: This function runs until user presses 'q' to quit
    """
    try:
        # Initialize detector
        detector = get_detector()
        
        # Create output directory for saved frames
        if save_frames:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            logger.info(f"Saving frames to {OUTPUT_DIR}")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open webcam with ID {camera_id}")
            return
        
        logger.info("Starting webcam mode. Press 'q' to quit.")
        
        frame_count = 0
        save_time = time.time()
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture image from webcam")
                break
            
            # Convert to RGB for model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect liveness
            result = detect_from_image(rgb_frame, threshold)
            
            # Process results
            color = (0, 255, 0) if result["is_live"] else (0, 0, 255)
            cv2.putText(
                frame,
                f"{result['status']} - {result['live_probability']:.4f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            
            # Display the frame if requested
            if display:
                cv2.imshow('Liveness Detection', frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Save frames if requested
            if save_frames:
                current_time = time.time()
                if current_time - save_time >= 1.0:  # Save one frame per second
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{OUTPUT_DIR}/frame_{timestamp}_{result['status']}_{result['live_probability']:.4f}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Frame saved: {filename}")
                    save_time = current_time
            
            # Print results to console
            frame_count += 1
            if frame_count % 10 == 0:  # Update console every 10 frames
                print(f"\rLiveness: {result['status']} - Probability: {result['live_probability']:.4f}", end="")
                sys.stdout.flush()
            
            # If not displaying, add a short delay to prevent high CPU usage
            if not display:
                time.sleep(0.05)
            
    except KeyboardInterrupt:
        logger.info("\nWebcam mode stopped by user")
    except Exception as e:
        logger.error(f"Error in webcam mode: {e}")
    
    finally:
        # Clean up
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if display:
            cv2.destroyAllWindows()
        print("\nWebcam session ended. Check the output directory for saved frames.") 