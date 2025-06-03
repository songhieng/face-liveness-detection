#!/usr/bin/env python3
"""
Example of how to integrate the Face Liveness Detection System
with another project without using the API.

This example shows how to:
1. Process an image file
2. Process a base64 encoded image
3. Batch process multiple images
4. Use webcam feed
5. Integrate with a simple face detection pipeline
"""

import os
import sys
import cv2
import numpy as np
import base64
from typing import List, Dict, Tuple

# Add the project root to the path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the integration module
from src.integration import (
    detect_from_file,
    detect_from_image,
    detect_from_base64,
    detect_from_batch,
    detect_from_webcam
)

def example_process_file():
    """Example of processing a single image file."""
    print("\n=== Example: Process Single Image File ===")
    
    # Path to your test image
    image_path = "path/to/your/test_image.jpg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"File {image_path} not found. Please update the path.")
        return
    
    # Process the image
    result = detect_from_file(image_path)
    
    # Print results
    print(f"Result for {image_path}:")
    print(f"  Is Live: {result['is_live']}")
    print(f"  Live Probability: {result['live_probability']:.6f}")
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")

def example_process_base64():
    """Example of processing a base64 encoded image."""
    print("\n=== Example: Process Base64 Image ===")
    
    # Load an image file and convert to base64
    image_path = "path/to/your/test_image.jpg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"File {image_path} not found. Please update the path.")
        return
    
    # Read the image and convert to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Process the base64 image
    result = detect_from_base64(base64_image)
    
    # Print results
    print(f"Result for base64 image:")
    print(f"  Is Live: {result['is_live']}")
    print(f"  Live Probability: {result['live_probability']:.6f}")
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")

def example_batch_process():
    """Example of batch processing multiple images."""
    print("\n=== Example: Batch Process Multiple Images ===")
    
    # List of image paths
    image_paths = [
        "path/to/your/image1.jpg",
        "path/to/your/image2.jpg",
        "path/to/your/image3.jpg"
    ]
    
    # Process all images
    results = detect_from_batch(image_paths)
    
    # Print results
    for i, result in enumerate(results):
        print(f"Result for image {i+1}:")
        print(f"  Is Live: {result['is_live']}")
        print(f"  Live Probability: {result['live_probability']:.6f}")
        print(f"  Status: {result['status']}")
        print(f"  Message: {result['message']}")
        print("")

def example_webcam():
    """Example of using webcam feed."""
    print("\n=== Example: Webcam Feed ===")
    print("Press 'q' to quit")
    
    # Start webcam detection
    detect_from_webcam(camera_id=0, threshold=0.5, display=True)

def example_face_detection_integration():
    """
    Example of integrating liveness detection with face detection.
    
    This demonstrates how to use the liveness detection as part of
    a face processing pipeline.
    """
    print("\n=== Example: Integration with Face Detection ===")
    
    # Path to your test image
    image_path = "path/to/your/test_image.jpg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"File {image_path} not found. Please update the path.")
        return
    
    # Load face detection model (using OpenCV's face detector for simplicity)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image from {image_path}")
        return
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        # Extract face region
        face_img = image[y:y+h, x:x+w]
        
        # Perform liveness detection on the face
        result = detect_from_image(face_img)
        
        # Draw bounding box
        color = (0, 255, 0) if result['is_live'] else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Add text with results
        cv2.putText(
            image,
            f"{result['status']} - {result['live_probability']:.4f}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        
        # Print results
        print(f"Face {i+1}:")
        print(f"  Is Live: {result['is_live']}")
        print(f"  Live Probability: {result['live_probability']:.6f}")
        print(f"  Status: {result['status']}")
        print(f"  Message: {result['message']}")
    
    # Display the result
    cv2.imshow('Face Liveness Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Run all examples."""
    # Uncomment the examples you want to run
    example_process_file()
    # example_process_base64()
    # example_batch_process()
    # example_webcam()
    # example_face_detection_integration()

if __name__ == "__main__":
    main() 