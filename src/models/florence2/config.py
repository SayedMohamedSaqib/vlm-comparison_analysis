# config.py - Corrected Version

import os
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Audio / Vosk configuration
SAMPLE_RATE = 16000
WAKE_WORD = "wake up"

# Cross-platform compatible path handling
def get_default_vosk_path():
    """Get default Vosk model path that works across platforms"""
    home = Path.home()
    default_path = home / "models" / "vosk-model-small-en-us-0.15"
    return str(default_path)

VOSK_MODEL_DIR = os.getenv("VOSK_MODEL_DIR", get_default_vosk_path())

# Processing throttles / tolerances
MIN_PROCESS_INTERVAL = 0.5  # seconds between processing runs
VALID_DETECTION_AGE = 2.0   # seconds to keep detections valid without refresh

# Processor image sizes
PROC_SIZE_REFERRING = (480, 480)
PROC_SIZE_DEFAULT = (224, 224)

# Tracker defaults (SimpleTracker fallback)
SIMPLETRACKER_MAX_AGE = 30
SIMPLETRACKER_IOU_THR = 0.3

# Semantic matching threshold
SEMANTIC_MATCH_THRESHOLD = 0.6  # Increased from 0.45 for better accuracy

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def validate_config():
    """Validate configuration settings and log warnings/errors"""
    errors = []
    warnings = []
    
    # Check Vosk model directory
    if not os.path.exists(VOSK_MODEL_DIR):
        errors.append(f"VOSK model directory not found: {VOSK_MODEL_DIR}")
    
    # Check sample rate
    if SAMPLE_RATE not in [8000, 16000, 22050, 44100]:
        warnings.append(f"Unusual sample rate: {SAMPLE_RATE}")
    
    # Check processing interval
    if MIN_PROCESS_INTERVAL < 0.1:
        warnings.append("MIN_PROCESS_INTERVAL very low, may cause performance issues")
    
    # Check image sizes
    if any(dim < 32 or dim > 1024 for dim in PROC_SIZE_DEFAULT + PROC_SIZE_REFERRING):
        warnings.append("Image processing sizes outside typical range (32-1024)")
    
    # Log results
    for error in errors:
        logger.error(error)
    for warning in warnings:
        logger.warning(warning)
    
    return errors, warnings

# Initialize logging
def setup_logging():
    """Setup logging configuration"""
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Auto-setup logging when module is imported
setup_logging()

# Validate configuration on import
errors, warnings = validate_config()
if errors:
    logger.critical(f"Configuration errors found: {len(errors)} errors")
else:
    logger.info("Configuration validation passed")