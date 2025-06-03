"""
Command-line interface for the face liveness detection system.
"""

import argparse
import logging

from src.core.app import LivenessDetectionApp
from utils.logging import setup_logger

logger = setup_logger("liveness_cli")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face Liveness Detection System")
    parser.add_argument(
        "path", 
        nargs="?", 
        help="Path to image or directory (optional, runs webcam mode if not provided)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
    
    # Run the application
    app = LivenessDetectionApp()
    app.run(args.path)

if __name__ == "__main__":
    main() 