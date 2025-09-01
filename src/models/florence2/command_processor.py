# command_processor.py - Corrected Version

import time
import json
import traceback
import logging
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import torch
from PIL import Image

from models import get_models
import utils
import state
import audio
import processor
import config

logger = logging.getLogger(__name__)

_last_process_time = 0

def validate_frame(frame_bgr_snapshot: np.ndarray) -> bool:
    """Validate input frame before processing"""
    if frame_bgr_snapshot is None:
        logger.error("Frame is None")
        return False
    
    if not isinstance(frame_bgr_snapshot, np.ndarray):
        logger.error("Frame is not a numpy array")
        return False
    
    if len(frame_bgr_snapshot.shape) != 3 or frame_bgr_snapshot.shape[2] != 3:
        logger.error(f"Invalid frame shape: {frame_bgr_snapshot.shape}")
        return False
    
    if frame_bgr_snapshot.size == 0:
        logger.error("Frame is empty")
        return False
    
    return True

def process_command(frame_bgr_snapshot: np.ndarray) -> None:
    """
    Record audio, transcribe, map to task, call Florence, post-process,
    and update state.latest_mask / state.latest_detections / state.latest_caption.
    Designed to be run in a background thread.
    """
    global _last_process_time

    try:
        state.processing_event.set()
        state.latest_caption = None

        # Validate input frame
        if not validate_frame(frame_bgr_snapshot):
            logger.error("Invalid frame provided to process_command")
            return

        # Throttle processing
        now = time.time()
        if now - _last_process_time < config.MIN_PROCESS_INTERVAL:
            logger.debug("Processing throttled")
            return
        _last_process_time = now

        H, W = frame_bgr_snapshot.shape[:2]
        logger.info(f"Processing frame: {W}x{H}")

        # Record audio
        logger.info("Recording audio (7s)...")
        try:
            audio_np = audio.record_audio(seconds=7)
        except Exception as e:
            logger.error(f"Audio recording failed: {e}")
            return

        # Transcribe with Whisper
        logger.info("Transcribing with Whisper...")
        try:
            models = get_models()
            whisper_processor = models["whisper_processor"]
            whisper_model = models["whisper_model"]

            wp_inputs = whisper_processor(
                audio_np,
                sampling_rate=config.SAMPLE_RATE,
                return_tensors="pt",
                language="en",
                task="transcribe",
            )

            # Move inputs to device
            wp_inputs = {
                k: (v.to(models["device"]) if hasattr(v, 'to') else v) 
                for k, v in wp_inputs.items()
            }

            with torch.inference_mode():
                gen_args = {
                    k: v for k, v in wp_inputs.items() 
                    if hasattr(v, 'shape') or hasattr(v, 'dtype')
                }
                generated_ids = whisper_model.generate(**gen_args)

            speech_command = whisper_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            logger.info(f"Recognized speech command: '{speech_command}'")

        except Exception as e:
            logger.error(f"Speech transcription failed: {e}")
            return

        # Map speech to task
        try:
            task_code = processor.map_speech_to_task_semantic(speech_command)
            logger.info(f"Mapped task code: '{task_code}'")
            
            if not task_code:
                task_code = processor.get_default_task()
                logger.info(f"Using default task: '{task_code}'")

        except Exception as e:
            logger.error(f"Task mapping failed: {e}")
            task_code = processor.get_default_task()

        # Choose processing size based on task
        if task_code in ["<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"]:
            proc_W, proc_H = config.PROC_SIZE_REFERRING
        else:
            proc_W, proc_H = config.PROC_SIZE_DEFAULT

        # Prepare image for Florence
        try:
            small_image = cv2.resize(frame_bgr_snapshot, (proc_W, proc_H))
            pil_image = Image.fromarray(cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return

        # Process with Florence-2
        try:
            florence_processor = models["florence_processor"]
            florence_model = models["florence_model"]

            fl_inputs = florence_processor(
                text=task_code, images=pil_image, return_tensors="pt"
            )

            # Prepare inputs for Florence model
            fl_payload = {}
            if fl_inputs.get("input_ids") is not None:
                fl_payload["input_ids"] = fl_inputs.get("input_ids").to(models["device"])
            if fl_inputs.get("attention_mask") is not None:
                fl_payload["attention_mask"] = fl_inputs.get("attention_mask").to(models["device"])
            if fl_inputs.get("pixel_values") is not None:
                fl_payload["pixel_values"] = fl_inputs.get("pixel_values").to(
                    models["device"], dtype=models["torch_dtype"]
                )

            # Generate output
            with torch.inference_mode():
                gen_ids = florence_model.generate(**fl_payload, max_new_tokens=512)

            generated_text = florence_processor.batch_decode(
                gen_ids, skip_special_tokens=False
            )[0]

            logger.debug(f"Florence raw output: {generated_text}")

        except Exception as e:
            logger.error(f"Florence processing failed: {e}")
            return

        # Post-process Florence output
        parsed_answer = None
        if hasattr(florence_processor, "post_process_generation"):
            try:
                parsed_answer = florence_processor.post_process_generation(
                    generated_text, task=task_code, image_size=(proc_W, proc_H)
                )
                logger.info(f"Parsed Florence output: {parsed_answer}")
            except Exception as e:
                logger.warning(f"post_process_generation failed: {e}")

        # Fallback JSON parsing
        if parsed_answer is None:
            try:
                maybe_json = json.loads(generated_text)
                if isinstance(maybe_json, dict):
                    parsed_answer = maybe_json
                    logger.info(f"Parsed JSON Florence output: {parsed_answer}")
            except Exception:
                state.latest_caption = generated_text.strip()

        # Update global state based on parsed_answer and task
        update_global_state(parsed_answer, task_code, W, H, proc_W, proc_H)

    except Exception as e:
        logger.error(f"Error in process_command: {e}")
        traceback.print_exc()
    finally:
        state.processing_event.clear()

def update_global_state(
    parsed_answer: Optional[Dict[str, Any]], 
    task_code: str, 
    W: int, H: int, 
    proc_W: int, proc_H: int
) -> None:
    """Update global state with processing results"""
    
    with state.mask_lock:
        state.latest_mask = None

    detections = None

    if isinstance(parsed_answer, dict):
        
        # Object detection tasks
        if task_code in ["<OD>", "<OPEN_VOCABULARY_DETECTION>", "<DENSE_REGION_CAPTION>"]:
            detections = utils.parse_bboxes(parsed_answer, W, H, proc_W=proc_W, proc_H=proc_H)
            state.latest_caption = (
                parsed_answer.get("text") or 
                parsed_answer.get("caption") or 
                state.latest_caption
            )

        # Segmentation tasks
        elif task_code in [
            "<REFERRING_EXPRESSION_SEGMENTATION>", 
            "<REGION_TO_SEGMENTATION>", 
            "<REGION_TO_CATEGORY>"
        ]:
            # Handle masks
            mask_candidate = (
                parsed_answer.get("mask") or 
                parsed_answer.get("masks") or 
                parsed_answer.get("segmentation")
            )
            
            if mask_candidate is not None:
                resized = utils._safe_resize_mask(mask_candidate, (H, W))
                if resized is not None:
                    with state.mask_lock:
                        state.latest_mask = resized
                    logger.info("Segmentation mask updated")

            # Also check for bounding boxes in segmentation tasks
            boxes = utils.parse_bboxes(parsed_answer, W, H, proc_W=proc_W, proc_H=proc_H)
            if boxes:
                detections = boxes

        # Caption and OCR tasks
        elif task_code in ["<CAPTION>", "<DETAILED_CAPTION>", "<OCR>"]:
            detections = utils.parse_bboxes(parsed_answer, W, H, proc_W=proc_W, proc_H=proc_H)
            state.latest_caption = (
                parsed_answer.get("text") or 
                state.latest_caption
            )

        # Default handling
        else:
            state.latest_caption = (
                parsed_answer.get("text") or 
                parsed_answer.get("caption") or 
                str(parsed_answer)
            )

    # Fallback caption handling
    if parsed_answer is None and state.latest_caption is None:
        state.latest_caption = "Processing completed"

    # Update detections state
    with state.detections_lock:
        if detections:
            # Ensure all detections have required fields
            for det in detections:
                det.setdefault("score", 0.95)
                det.setdefault("label", "object")
            
            state.latest_detections = detections
            state.detections_ts = time.time()
            
            # Clear mask if we have detections
            if state.latest_mask is not None:
                with state.mask_lock:
                    state.latest_mask = None
                    
            logger.info(f"Updated detections: {len(detections)} objects")
        else:
            # Only clear detections if we don't have a mask
            if state.latest_mask is None:
                state.latest_detections = None
                state.detections_ts = time.time()