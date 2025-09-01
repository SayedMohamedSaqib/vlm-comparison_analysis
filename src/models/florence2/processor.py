# processor.py - Corrected Version

import logging
import torch
from models import get_models
from sentence_transformers import util
from typing import Optional
import config

logger = logging.getLogger(__name__)

# CORRECTED: Proper Florence-2 task prompts mapping with valid task codes
task_prompts = {
    "<OD>": [
        "detect objects", 
        "find items in the image", 
        "what objects are present",
        "show me objects",
        "identify things"
    ],
    "<CAPTION>": [
        "describe the image", 
        "give me a caption", 
        "explain what's happening here",
        "what do you see",
        "describe this"
    ],
    "<DETAILED_CAPTION>": [
        "describe the image in detail", 
        "give detailed caption",
        "detailed description",
        "comprehensive description"
    ],
    "<REFERRING_EXPRESSION_SEGMENTATION>": [
        "find the green car", 
        "where is the red cup", 
        "highlight the wine glass",
        "segment the specified object",
        "find and segment"
    ],
    "<REGION_TO_SEGMENTATION>": [
        "segment the blue cup", 
        "highlight the cat", 
        "mask the dog",
        "segment this region",
        "create mask"
    ],
    "<DENSE_REGION_CAPTION>": [
        "describe all regions", 
        "caption each area in image", 
        "breakdown the scene",
        "detailed region analysis"
    ],
    "<OCR>": [
        "read text", 
        "extract words", 
        "find all labels",
        "what does the text say",
        "read the signs"
    ],
    "<OPEN_VOCABULARY_DETECTION>": [
        "detect any object", 
        "find unusual items",
        "open vocabulary detection",
        "detect anything"
    ],
    "<REGION_TO_CATEGORY>": [
        "segment objects", 
        "mask items",
        "categorize regions",
        "classify segments"
    ],
}

def map_speech_to_task_semantic(speech_text: str, threshold: Optional[float] = None) -> str:
    """
    Map speech command to Florence-2 task using semantic similarity
    
    Args:
        speech_text: The recognized speech command
        threshold: Minimum similarity threshold (uses config default if None)
    
    Returns:
        Florence-2 task code or empty string if no match above threshold
    """
    speech_text = (speech_text or "").lower().strip()
    if not speech_text:
        logger.warning("Empty speech text provided")
        return ""
    
    # Use configurable threshold
    if threshold is None:
        threshold = getattr(config, 'SEMANTIC_MATCH_THRESHOLD', 0.6)
    
    try:
        models = get_models()
        st_model = models["st_model"]
        
        # Encode the speech command
        speech_emb = st_model.encode(speech_text, convert_to_tensor=True)
        
        best_task = ""
        best_score = -1.0
        
        # Compare against all task prompts
        for task, prompts in task_prompts.items():
            try:
                prompt_embs = st_model.encode(prompts, convert_to_tensor=True)
                scores = util.cos_sim(speech_emb, prompt_embs)
                max_score = float(torch.max(scores).item())
                
                if max_score > best_score:
                    best_score = max_score
                    best_task = task
                    
            except Exception as e:
                logger.error(f"Error processing task {task}: {e}")
                continue
        
        # Only return task if above threshold
        if best_score >= threshold:
            logger.info(f"Mapped '{speech_text}' to task '{best_task}' with score {best_score:.3f}")
            return best_task
        else:
            logger.warning(f"No task match above threshold {threshold} for '{speech_text}' (best: {best_score:.3f})")
            return ""
            
    except Exception as e:
        logger.error(f"Error in semantic mapping: {e}")
        return ""

def get_available_tasks():
    """Get list of available Florence-2 tasks"""
    return list(task_prompts.keys())

def get_task_examples(task_code: str):
    """Get example prompts for a specific task"""
    return task_prompts.get(task_code, [])

def validate_task_code(task_code: str) -> bool:
    """Validate if a task code is supported"""
    return task_code in task_prompts

def get_default_task() -> str:
    """Get default task when no specific match is found"""
    return "<CAPTION>"  # Default to image captioning