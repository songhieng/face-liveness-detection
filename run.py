#!/usr/bin/env python3
"""
Face Liveness Detection System Runner

This script provides a convenient way to run different components of the liveness detection system.

Usage:
    python run.py [command] [options]

Commands:
    api         - Start the FastAPI server
    detect      - Run liveness detection on an image or directory
    webcam      - Run liveness detection on webcam feed
"""

import sys
from utils.runner import main

if __name__ == "__main__":
    sys.exit(main()) 