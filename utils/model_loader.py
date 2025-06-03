"""
Model loading utilities for the face liveness detection system.
"""

import os
import sys
import torch
from typing import Dict
from utils.logging import setup_logger

logger = setup_logger("model_loader")

def load_model(model: torch.nn.Module, state_dict: Dict) -> None:
    """
    Load pretrained weights into model.
    
    Args:
        model: PyTorch model to load weights into
        state_dict: Dictionary containing model weights
    """
    own_state = model.state_dict()
    for name, param in state_dict.items():
        realname = name.replace('module.', '')
        if realname in own_state:
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            try:
                own_state[realname].copy_(param)
            except Exception as e:
                logger.warning(f"Error while copying parameter {realname}: {e}") 