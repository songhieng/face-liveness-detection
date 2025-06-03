"""
FastAPI application for the face liveness detection system.
"""

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as detect_router
from utils.logging import setup_logger
import config.settings as settings

logger = setup_logger("liveness_api")

# Create FastAPI app
app = FastAPI(
    title="Face Liveness Detection API",
    description="API for detecting real (live) vs fake (spoof) faces in images",
    version=settings.__version__ if hasattr(settings, "__version__") else "1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root router
root_router = APIRouter()

@root_router.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Face Liveness Detection API is running"}

@root_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": app.version,
    }

# Include routers
app.include_router(root_router)
app.include_router(detect_router)

@app.on_event("startup")
async def startup_event():
    """Initialize resources when the API starts."""
    logger.info("Starting Face Liveness Detection API")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the API shuts down."""
    logger.info("Shutting down Face Liveness Detection API") 