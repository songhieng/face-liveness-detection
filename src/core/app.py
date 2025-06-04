"""
CLI application for running liveness detection.
"""

import os
import sys
import cv2
import logging
import time
from typing import Optional
from datetime import datetime

from src.core.detector import LivenessDetector
from utils.logging import setup_logger
from config.settings import LIVENESS_THRESHOLD, PROJECT_ROOT

logger = setup_logger("liveness_app")

# Ensure output directory exists
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "webcam_frames")

class LivenessDetectionApp:
    """Application for running liveness detection on images or webcam."""
    
    def __init__(self):
        """Initialize the application."""
        self.detector = None
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def initialize_detector(self) -> None:
        """Initialize the detector if it hasn't been initialized yet."""
        if self.detector is None:
            logger.info("Initializing liveness detector...")
            try:
                self.detector = LivenessDetector()
                logger.info("Detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize detector: {e}")
                sys.exit(1)
    
    def process_image(self, image_path: str) -> None:
        """
        Process a single image and print results.
        
        Args:
            image_path: Path to the image file
        """
        try:
            logger.info(f"Processing image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image from {image_path}")
                return
            
            # Convert to RGB (OpenCV uses BGR)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect liveness
            prob = self.detector.detect(rgb)[0]  # [p_spoof, p_live]
            
            # Print results
            is_live = prob[1] > LIVENESS_THRESHOLD
            label = "REAL FACE (Live)" if is_live else "FAKE FACE (Spoof)"
            
            print(f"\nImage: {image_path}")
            print(f"  Probability of being real (live): {prob[1]:.6f}")
            print(f"  Probability of being fake (spoof): {prob[0]:.6f}")
            print(f"  Result: {label}")
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
    
    def process_directory(self, directory_path: str) -> None:
        """
        Process all supported images in a directory.
        
        Args:
            directory_path: Path to the directory containing images
        """
        try:
            # Get all supported image files
            supported_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = []
            
            for file in sorted(os.listdir(directory_path)):
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_ext:
                    image_files.append(os.path.join(directory_path, file))
            
            if not image_files:
                logger.warning(f"No supported images found in folder: {directory_path}")
                return
            
            logger.info(f"Found {len(image_files)} image(s) in folder: {directory_path}")
            
            # Process each image
            for image_path in image_files:
                self.process_image(image_path)
                
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
    
    def run_webcam(self) -> None:
        """Run liveness detection on webcam feed and display video."""
        logger.info("Starting webcam mode...")
        
        # Create output directory for saved frames
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Saving frames to {OUTPUT_DIR}")
        
        try:
            # Open webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Could not open webcam")
                return
            
            logger.info("Running liveness detection on webcam feed. Press 'q' to quit.")
            
            frame_count = 0
            start_time = time.time()
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
                prob = self.detector.detect(rgb_frame)[0]  # [p_spoof, p_live]
                is_live = prob[1] > LIVENESS_THRESHOLD
                live_prob = prob[1]
                result = "REAL FACE (Live)" if is_live else "FAKE FACE (Spoof)"
                
                # Overlay text on frame
                color = (0, 255, 0) if is_live else (0, 0, 255)
                cv2.putText(
                    frame,
                    f"{result} - {live_prob:.4f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                
                # Display the frame
                cv2.imshow('Liveness Detection', frame)
                
                # Save the frame to a file every second
                current_time = time.time()
                if current_time - save_time >= 1.0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{OUTPUT_DIR}/frame_{timestamp}_{result.replace(' ', '_')}_{live_prob:.4f}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Frame saved: {filename}")
                    save_time = current_time
                
                # Print results to console
                frame_count += 1
                if frame_count % 10 == 0:  # Update console every 10 frames
                    print(f"\rLiveness: {result} - Probability: {live_prob:.4f}", end="")
                    sys.stdout.flush()
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            logger.info("\nWebcam mode stopped by user")
        except Exception as e:
            logger.error(f"Error in webcam mode: {e}")
        
        finally:
            # Clean up
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            print("\nWebcam session ended. Check the output directory for saved frames.")
    
    def run(self, path: Optional[str] = None) -> None:
        """
        Run the application with the given path.
        
        Args:
            path: Path to image or directory. If None, runs webcam mode.
        """
        self.initialize_detector()
        
        if path is None:
            # No path provided, run webcam mode
            self.run_webcam()
        elif os.path.isdir(path):
            # Path is a directory
            self.process_directory(path)
        elif os.path.isfile(path):
            # Path is a file
            self.process_image(path)
        else:
            logger.error(f"Invalid path: {path}") 