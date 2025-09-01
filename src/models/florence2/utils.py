# utils.py - Corrected Version

import numpy as np
import base64
from PIL import Image
import io
import cv2
from typing import Optional, Dict, Any, List, Union, Tuple
import logging

logger = logging.getLogger(__name__)

def _decode_mask_candidate(x: Any) -> Optional[np.ndarray]:
    """
    Decode various mask formats to numpy array.
    
    Args:
        x: Mask data in various formats (numpy array, base64 string, PIL Image, etc.)
    
    Returns:
        Decoded mask as numpy array or None if failed
    """
    if x is None:
        return None

    # Try direct numpy array conversion
    try:
        arr = np.asarray(x)
        if arr.size > 0:
            return arr
    except Exception:
        pass

    # Try base64 decoding
    try:
        if isinstance(x, str) and (x.startswith("data:") or _is_base64(x)):
            if x.startswith("data:"):
                # Extract base64 part from data URL
                b64 = x.split(",", 1)[1]
            else:
                b64 = x
            
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("L")
            return np.array(img)
    except Exception as e:
        logger.debug(f"Base64 decode failed: {e}")
        pass

    return None

def _is_base64(s: str) -> bool:
    """Check if string is valid base64"""
    if len(s) % 4 != 0:
        return False
    return all(c.isalnum() or c in "+/=" for c in s[:50])

def _safe_resize_mask(mask: Any, target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Safely resize mask to target dimensions.
    
    Args:
        mask: Input mask in various formats
        target_hw: Target (height, width)
    
    Returns:
        Resized mask or None if failed
    """
    try:
        mask = _decode_mask_candidate(mask)
        if mask is None:
            return None

        # Handle multi-channel masks
        if mask.ndim == 3:
            mask = mask[..., 0]
        
        # Ensure proper data type
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8) * 255

        # Resize using nearest neighbor to preserve binary nature
        resized = cv2.resize(
            mask, 
            (target_hw[1], target_hw[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        return resized
        
    except Exception as e:
        logger.error(f"Mask resize failed: {e}")
        return None

def wrap_text(text: str, max_chars: int = 40) -> List[str]:
    """
    Wrap text into lines of specified maximum length.
    
    Args:
        text: Input text to wrap
        max_chars: Maximum characters per line
    
    Returns:
        List of wrapped text lines
    """
    if not text:
        return []
    
    lines = []
    words = text.split()
    current_line = ""
    
    for word in words:
        # If adding this word would exceed limit, start new line
        if current_line and len(current_line) + len(word) + 1 > max_chars:
            lines.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
    
    # Add final line if not empty
    if current_line:
        lines.append(current_line)
    
    return lines

def _extract_bbox_candidates(parsed_answer: Dict[str, Any]) -> Optional[List[Any]]:
    """Extract bounding box candidates from parsed answer"""
    for key in ("bboxes", "boxes", "bbox", "detections", "regions", "items"):
        if key in parsed_answer and parsed_answer[key]:
            return parsed_answer[key]
    return None

def _normalize_single_bbox(
    entry: Any, 
    labels: Optional[List[str]], 
    index: int, 
    W: int, H: int, 
    proc_W: Optional[int] = None, 
    proc_H: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Normalize a single bbox entry to standard format.
    
    Returns:
        Normalized bbox dict or None if invalid
    """
    try:
        label = ""
        score = None
        
        # Handle different bbox formats
        if isinstance(entry, dict):
            # Format 1: {bbox: [x1,y1,x2,y2], label: "...", score: 0.9}
            if "bbox" in entry and isinstance(entry["bbox"], (list, tuple)) and len(entry["bbox"]) >= 4:
                x1, y1, x2, y2 = entry["bbox"][:4]
                label = entry.get("label") or entry.get("text") or entry.get("caption") or ""
                score = entry.get("score") or entry.get("confidence")
            
            # Format 2: {x: 10, y: 20, w: 50, h: 30}
            elif all(k in entry for k in ("x", "y", "w", "h")):
                x1 = entry["x"]
                y1 = entry["y"]  
                x2 = x1 + entry["w"]
                y2 = y1 + entry["h"]
                label = entry.get("label", "")
                score = entry.get("score")
            
            # Format 3: {box: [x1,y1,x2,y2]}
            elif "box" in entry and isinstance(entry["box"], (list, tuple)) and len(entry["box"]) >= 4:
                x1, y1, x2, y2 = entry["box"][:4]
                label = entry.get("label", "")
                score = entry.get("score")
            else:
                return None
                
        # Format 4: Direct list/tuple [x1, y1, x2, y2]
        elif isinstance(entry, (list, tuple)) and len(entry) >= 4:
            x1, y1, x2, y2 = entry[:4]
            label = (labels[index] if labels and index < len(labels) else "") or ""
            score = None
        else:
            return None

        # Convert to float
        x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
        
        # Coordinate normalization logic
        if _is_normalized_coords(x1f, y1f, x2f, y2f):
            # Normalized coordinates [0,1]
            x1p, x2p = x1f * W, x2f * W
            y1p, y2p = y1f * H, y2f * H
            
        elif proc_W and proc_H and _is_processed_coords(x1f, y1f, x2f, y2f, proc_W, proc_H):
            # Coordinates in processing resolution
            sx, sy = float(W) / float(proc_W), float(H) / float(proc_H)
            x1p, x2p = x1f * sx, x2f * sx
            y1p, y2p = y1f * sy, y2f * sy
            
        else:
            # Assume absolute coordinates
            x1p, x2p, y1p, y2p = x1f, x2f, y1f, y2f

        # Convert to integer pixel coordinates
        x1i = max(0, min(W - 1, int(round(x1p))))
        y1i = max(0, min(H - 1, int(round(y1p))))
        x2i = max(0, min(W - 1, int(round(x2p))))
        y2i = max(0, min(H - 1, int(round(y2p))))

        # Validate bbox
        if x2i <= x1i or y2i <= y1i:
            logger.debug(f"Invalid bbox dimensions: ({x1i},{y1i}) to ({x2i},{y2i})")
            return None
        
        # Validate minimum size
        if (x2i - x1i) < 2 or (y2i - y1i) < 2:
            logger.debug(f"Bbox too small: {x2i-x1i}x{y2i-y1i}")
            return None

        # Set default values
        if score is None:
            score = 0.95

        return {
            "bbox": [x1i, y1i, x2i, y2i],
            "label": str(label),
            "score": float(score)
        }
        
    except Exception as e:
        logger.debug(f"Error normalizing bbox at index {index}: {e}")
        return None

def _is_normalized_coords(x1: float, y1: float, x2: float, y2: float) -> bool:
    """Check if coordinates are normalized [0,1]"""
    return (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 
            0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0)

def _is_processed_coords(x1: float, y1: float, x2: float, y2: float, 
                        proc_W: int, proc_H: int) -> bool:
    """Check if coordinates are in processing resolution"""
    max_coord = max(x1, y1, x2, y2)
    max_proc_dim = max(proc_W, proc_H)
    return max_coord <= max_proc_dim + 1

def parse_bboxes(
    parsed_answer: Dict[str, Any], 
    W: int, H: int, 
    proc_W: Optional[int] = None, 
    proc_H: Optional[int] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Parse and normalize bounding boxes from Florence-2 output.
    
    Args:
        parsed_answer: Parsed Florence-2 response
        W, H: Target image dimensions
        proc_W, proc_H: Processing image dimensions (optional)
    
    Returns:
        List of normalized bbox dicts or None if no valid boxes found
    """
    if not isinstance(parsed_answer, dict):
        logger.debug("Parsed answer is not a dictionary")
        return None

    # Handle nested single-key dictionaries
    if len(parsed_answer) == 1:
        single_value = list(parsed_answer.values())[0]
        if isinstance(single_value, dict):
            parsed_answer = single_value

    # Extract bbox candidates
    candidate = _extract_bbox_candidates(parsed_answer)
    if candidate is None:
        logger.debug("No bbox candidates found in parsed answer")
        return None

    # Extract labels if available
    labels = parsed_answer.get("labels")

    # Process each bbox candidate
    normalized_bboxes = []
    for i, entry in enumerate(candidate):
        normalized = _normalize_single_bbox(entry, labels, i, W, H, proc_W, proc_H)
        if normalized is not None:
            normalized_bboxes.append(normalized)

    if not normalized_bboxes:
        logger.debug("No valid bboxes after normalization")
        return None

    logger.info(f"Successfully parsed {len(normalized_bboxes)} bounding boxes")
    return normalized_bboxes

def validate_bbox(bbox: Dict[str, Any]) -> bool:
    """Validate a normalized bbox dictionary"""
    required_keys = ["bbox", "label", "score"]
    
    if not all(key in bbox for key in required_keys):
        return False
    
    if not isinstance(bbox["bbox"], (list, tuple)) or len(bbox["bbox"]) != 4:
        return False
    
    try:
        x1, y1, x2, y2 = bbox["bbox"]
        if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
            return False
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        score = float(bbox["score"])
        if not 0.0 <= score <= 1.0:
            return False
            
    except (ValueError, TypeError):
        return False
    
    return True