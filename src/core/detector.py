"""
Core implementation of the face liveness detector.
"""

import os
import torch
import numpy as np
from typing import Union, List, Tuple, Optional
from PIL import Image
import torchvision

from utils.logging import setup_logger
from utils.model_loader import load_model
from config.settings import MODEL_CHECKPOINT_PATH, IMAGE_SIZE, NUM_CLASSES
from src.models.aenet import AENet
from src.models.detector import CelebASpoofDetector

logger = setup_logger("liveness_detector")

class LivenessDetector:
    """
    Face liveness detection using the CelebA-Spoof model.
    
    This class provides methods to detect whether a face is real (live) or fake (spoof).
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the liveness detector.
        
        Args:
            model_path: Optional path to the model checkpoint file. If None, uses default path.
            device: Optional device to use ('cuda' or 'cpu'). If None, auto-selects based on availability.
        """
        # Set parameters
        self.num_class = NUM_CLASSES
        self.image_size = IMAGE_SIZE
        
        # Initialize model
        self.net = AENet(num_classes=self.num_class)
        
        # Set device (cuda or cpu)
        self.device = self._set_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load model checkpoint
        self._load_checkpoint(model_path)
        
        # Set up image transformation
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_size, self.image_size)),
            torchvision.transforms.ToTensor(),
        ])
        
        # Set model to evaluation mode
        self.net.to(self.device)
        self.net.eval()
    
    def _set_device(self, device: Optional[str]) -> torch.device:
        """
        Set the device to use for model inference.
        
        Args:
            device: Optional device specification ('cuda' or 'cpu')
            
        Returns:
            torch.device: The device to use
        """
        if device is not None:
            return torch.device(device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _load_checkpoint(self, model_path: Optional[str] = None) -> None:
        """
        Load model weights from checkpoint.
        
        Args:
            model_path: Optional path to the model checkpoint file
        """
        if model_path is None:
            model_path = MODEL_CHECKPOINT_PATH
        
        try:
            logger.info(f"Loading checkpoint from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            load_model(self.net, checkpoint['state_dict'])
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Failed to load model checkpoint: {e}") from e
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image as numpy array (RGB format)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            processed_data = Image.fromarray(image)
            processed_data = self.transform(processed_data)
            return processed_data
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Failed to preprocess image: {e}") from e
    
    def detect(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Detect liveness in one or more images.
        
        Args:
            images: Single image as numpy array (RGB) or list of images
            
        Returns:
            np.ndarray: Array of probabilities, shape [N, 2] where each row is [p_spoof, p_live]
        """
        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]
        
        try:
            # Preprocess all images
            preprocessed_list = [self.preprocess_image(img) for img in images]
            
            # Stack tensors and move to device
            data = torch.stack(preprocessed_list, dim=0)
            input_tensor = data.view(-1, 3, data.size(2), data.size(3)).to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.net(input_tensor).detach()
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Liveness detection failed: {e}")
            raise RuntimeError(f"Failed to perform liveness detection: {e}") from e
    
    def is_live(self, image: np.ndarray, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Check if a face is live (real) based on a threshold.
        
        Args:
            image: Input image as numpy array (RGB format)
            threshold: Probability threshold to classify as live (default: 0.5)
            
        Returns:
            Tuple[bool, float]: (is_live, live_probability)
        """
        probs = self.detect(image)[0]  # [p_spoof, p_live]
        live_prob = probs[1]
        return live_prob > threshold, live_prob 