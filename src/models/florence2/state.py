# state.py - Corrected Version

import threading
import queue
import time
from typing import Optional, List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Frame buffer
latest_frame: Optional[np.ndarray] = None
frame_lock = threading.Lock()

# Mask / detection state  
latest_mask: Optional[np.ndarray] = None
mask_lock = threading.Lock()

# Latest caption text
latest_caption: Optional[str] = None

# Events & control
processing_event = threading.Event()
stop_event = threading.Event()

# Detections and timestamp
latest_detections: Optional[List[Dict[str, Any]]] = None
detections_lock = threading.Lock()
detections_ts: float = 0.0

# Audio queue (raw int16 bytes pieces)
audio_queue: queue.Queue = queue.Queue()

# Performance monitoring
_stats = {
    'frames_processed': 0,
    'detections_count': 0,
    'processing_time_avg': 0.0,
    'last_processing_time': 0.0
}
stats_lock = threading.Lock()

def get_stats() -> Dict[str, Any]:
    """Get current performance statistics"""
    with stats_lock:
        return _stats.copy()

def update_stats(processing_time: float, detection_count: int = 0) -> None:
    """Update performance statistics"""
    with stats_lock:
        _stats['frames_processed'] += 1
        _stats['detections_count'] = detection_count
        _stats['last_processing_time'] = processing_time
        
        # Update rolling average processing time
        alpha = 0.1  # smoothing factor
        _stats['processing_time_avg'] = (
            alpha * processing_time + 
            (1 - alpha) * _stats['processing_time_avg']
        )

def reset_stats() -> None:
    """Reset performance statistics"""
    with stats_lock:
        _stats.update({
            'frames_processed': 0,
            'detections_count': 0,
            'processing_time_avg': 0.0,
            'last_processing_time': 0.0
        })

def get_state_summary() -> Dict[str, Any]:
    """Get summary of current state for debugging"""
    with frame_lock, mask_lock, detections_lock:
        return {
            'has_frame': latest_frame is not None,
            'frame_shape': latest_frame.shape if latest_frame is not None else None,
            'has_mask': latest_mask is not None,
            'mask_shape': latest_mask.shape if latest_mask is not None else None,
            'has_detections': latest_detections is not None,
            'detection_count': len(latest_detections) if latest_detections else 0,
            'has_caption': latest_caption is not None,
            'caption_length': len(latest_caption) if latest_caption else 0,
            'processing_active': processing_event.is_set(),
            'stop_requested': stop_event.is_set(),
            'detections_age': time.time() - detections_ts if detections_ts > 0 else float('inf'),
            'audio_queue_size': audio_queue.qsize(),
            'stats': get_stats()
        }

def clear_all_state() -> None:
    """Clear all state variables (useful for testing or reset)"""
    global latest_frame, latest_mask, latest_caption, latest_detections, detections_ts
    
    with frame_lock:
        latest_frame = None
    
    with mask_lock:
        latest_mask = None
    
    with detections_lock:
        latest_detections = None
        detections_ts = 0.0
    
    latest_caption = None
    
    # Clear events
    processing_event.clear()
    # Note: Don't clear stop_event as it might be intentionally set
    
    # Clear audio queue
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except:
            break
    
    reset_stats()
    logger.info("All state cleared")

def is_processing_active() -> bool:
    """Check if processing is currently active"""
    return processing_event.is_set()

def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested"""
    return stop_event.is_set()

def wait_for_processing_complete(timeout: float = 30.0) -> bool:
    """Wait for current processing to complete
    
    Args:
        timeout: Maximum time to wait in seconds
    
    Returns:
        True if processing completed, False if timeout
    """
    start_time = time.time()
    while processing_event.is_set():
        if time.time() - start_time > timeout:
            logger.warning(f"Processing did not complete within {timeout}s timeout")
            return False
        time.sleep(0.1)
    return True