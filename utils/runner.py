"""
Runner utilities for the face liveness detection system.
"""

import os
import sys
import subprocess
import argparse
from typing import Optional, List, Any, Dict, Callable

from utils.logging import setup_logger

logger = setup_logger("runner")

def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> int:
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        reload: Whether to enable auto-reload for development
        
    Returns:
        int: Return code
    """
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn is not installed. Please install it with: pip install uvicorn[standard]")
        return 1
    
    reload_arg = "--reload" if reload else ""
    cmd = f"uvicorn src.api.app:app --host {host} --port {port} {reload_arg}"
    logger.info(f"Starting API server with command: {cmd}")
    
    try:
        # Use subprocess to run the uvicorn command
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error running API server: {e}")
        return 1

def run_detect(path: Optional[str] = None, verbose: bool = False) -> int:
    """
    Run liveness detection on an image or directory.
    
    Args:
        path: Path to image or directory
        verbose: Whether to enable verbose logging
        
    Returns:
        int: Return code
    """
    try:
        from src.core.cli import main
        
        # Set up arguments
        sys.argv = [sys.argv[0]]
        if path:
            sys.argv.append(path)
        if verbose:
            sys.argv.append("--verbose")
        
        # Run the main function
        main()
        return 0
    except Exception as e:
        logger.error(f"Error running detection: {e}")
        return 1

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Face Liveness Detection System Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start the FastAPI server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Run liveness detection on an image or directory")
    detect_parser.add_argument("path", nargs="?", help="Path to image or directory (optional, runs webcam mode if not provided)")
    detect_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # Webcam command
    webcam_parser = subparsers.add_parser("webcam", help="Run liveness detection on webcam feed")
    webcam_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

def main() -> int:
    """
    Main entry point for the runner.
    
    Returns:
        int: Return code
    """
    args = parse_arguments()
    
    if args.command == "api":
        return run_api(host=args.host, port=args.port, reload=args.reload)
    elif args.command == "detect":
        return run_detect(path=args.path, verbose=args.verbose)
    elif args.command == "webcam":
        return run_detect(verbose=args.verbose)
    else:
        parse_arguments()  # Show help
        return 1 