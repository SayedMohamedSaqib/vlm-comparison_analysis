# models.py - Corrected Version

import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global models dictionary
_MODELS: Dict[str, Any] = {}

def load_models(device: Optional[str] = None, torch_dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
    """
    Loads Whisper, Florence, and SentenceTransformer and stores in module-level _MODELS.
    Returns the dict of models.
    """
    if _MODELS:
        logger.info("Models already loaded, returning cached models")
        return _MODELS

    # Device selection with validation
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        else:
            logger.warning("CUDA not available, using CPU (this will be slow)")

    # Data type selection
    if torch_dtype is None:
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    logger.info(f"Loading models on {device} with dtype {torch_dtype}")

    try:
        # Import heavy libraries here to avoid import time overhead
        from transformers import (
            WhisperProcessor, 
            WhisperForConditionalGeneration, 
            AutoProcessor, 
            AutoModelForCausalLM
        )
        from sentence_transformers import SentenceTransformer

        # Load Whisper
        logger.info("Loading Whisper model...")
        try:
            whisper_model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-small"
            ).to(device)
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise

        # Load Florence-2
        logger.info("Loading Florence-2 model...")
        try:
            florence_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base", 
                torch_dtype=torch_dtype, 
                trust_remote_code=True
            ).to(device)
            florence_processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", 
                trust_remote_code=True
            )
            logger.info("Florence-2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Florence-2: {e}")
            raise

        # Load Sentence Transformer
        logger.info("Loading Sentence Transformer...")
        try:
            st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            logger.info("Sentence Transformer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer: {e}")
            raise

        # Store models in global dictionary
        _MODELS.update({
            "device": device,
            "torch_dtype": torch_dtype,
            "whisper_model": whisper_model,
            "whisper_processor": whisper_processor,
            "florence_model": florence_model,
            "florence_processor": florence_processor,
            "st_model": st_model,
        })

        logger.info("All models loaded successfully")
        return _MODELS

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Clean up partial loads
        cleanup_models()
        raise

def get_models() -> Dict[str, Any]:
    """Get loaded models dictionary"""
    if not _MODELS:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    return _MODELS

def cleanup_models() -> None:
    """Clean up loaded models to free memory"""
    global _MODELS
    
    if not _MODELS:
        logger.info("No models to clean up")
        return
        
    try:
        device = _MODELS.get("device", "cpu")
        
        # Move models to CPU to free GPU memory
        for key, model in _MODELS.items():
            if hasattr(model, 'to') and key not in ["device", "torch_dtype"]:
                try:
                    model.to('cpu')
                    logger.debug(f"Moved {key} to CPU")
                except Exception as e:
                    logger.warning(f"Failed to move {key} to CPU: {e}")
        
        # Clear CUDA cache if we were using GPU
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
                
        # Clear the models dictionary
        _MODELS.clear()
        logger.info("Models cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during model cleanup: {e}")
        # Force clear the dictionary even if cleanup failed
        _MODELS.clear()

def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage statistics"""
    stats = {}
    
    # PyTorch memory stats
    if torch.cuda.is_available():
        stats['cuda_allocated'] = torch.cuda.memory_allocated() / (1024**3)  # GB
        stats['cuda_reserved'] = torch.cuda.memory_reserved() / (1024**3)   # GB
        stats['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    
    # Model loading status
    stats['models_loaded'] = len(_MODELS) > 0
    stats['model_count'] = len([k for k in _MODELS.keys() if not k.startswith(('device', 'torch_dtype'))])
    
    return stats

def is_models_loaded() -> bool:
    """Check if models are loaded"""
    return len(_MODELS) > 0

def get_device() -> str:
    """Get the device models are loaded on"""
    return _MODELS.get("device", "unknown")