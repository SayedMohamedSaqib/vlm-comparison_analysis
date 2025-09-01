import cv2
import threading
import time
import traceback
import json
import logging
import atexit

import config
import state
import models
import audio
import tracker
import video
import command_processor

logger = logging.getLogger(__name__)

def setup_cleanup():
    """Register cleanup functions to run on exit"""
    atexit.register(cleanup_resources)

def cleanup_resources():
    """Clean up all resources"""
    logger.info("Starting resource cleanup...")
    state.stop_event.set()
    try:
        models.cleanup_models()
    except Exception as e:
        logger.error(f"Error cleaning up models: {e}")
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"Error destroying OpenCV windows: {e}")
    logger.info("Resource cleanup complete")

def main():
    """Main application entry point"""
    setup_cleanup()
    cap = None
    stream = None
    video_thread = None
    try:
        logger.info("Starting application...")
        logger.info("Loading models (this may take a while)...")
        models.load_models()
        logger.info("Models loaded successfully")

        logger.info("Initializing audio system...")
        recognizer = audio.init_vosk()
        stream = audio.start_audio_stream()
        logger.info("Audio system initialized, listening for wake word...")

        logger.info("Opening webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam (index 0)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        logger.info("Webcam opened successfully")

        logger.info("Initializing object tracker...")
        tracker_inst = tracker.init_tracker()
        logger.info("Object tracker initialized")

        logger.info("Starting video capture thread...")
        video_thread = threading.Thread(
            target=video.video_loop,
            args=(cap, tracker_inst),
            daemon=True,
            name="VideoThread"
        )
        video_thread.start()
        logger.info("Video thread started")

        logger.info(f"Say '{config.WAKE_WORD}' to trigger processing")
        main_audio_loop(recognizer)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        traceback.print_exc()
    finally:
        logger.info("Shutting down application...")
        state.stop_event.set()
        if stream:
            try:
                stream.stop()
                stream.close()
                logger.info("Audio stream stopped")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
        if video_thread and video_thread.is_alive():
            logger.info("Waiting for video thread to finish...")
            video_thread.join(timeout=5.0)
            if video_thread.is_alive():
                logger.warning("Video thread did not shutdown gracefully within timeout")
            else:
                logger.info("Video thread finished")
        if cap:
            try:
                cap.release()
                logger.info("Webcam released")
            except Exception as e:
                logger.error(f"Error releasing webcam: {e}")
        cleanup_resources()
        logger.info("Shutdown complete")

def main_audio_loop(recognizer) -> None:
    """Main audio processing loop"""
    consecutive_errors = 0
    max_consecutive_errors = 10
    while not state.stop_event.is_set():
        try:
            audio_processed = process_audio_queue(recognizer)
            if audio_processed:
                consecutive_errors = 0
            else:
                time.sleep(0.01)
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Error in audio loop (#{consecutive_errors}): {e}")
            if consecutive_errors >= max_consecutive_errors:
                logger.critical("Too many consecutive errors in audio loop, stopping")
                state.stop_event.set()
                break
            time.sleep(0.1)

def process_audio_queue(recognizer) -> bool:
    """Process available audio data and detect wake words"""
    audio_processed = False
    while True:
        try:
            data = state.audio_queue.get_nowait()
            audio_processed = True
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text = json.loads(result).get("text", "")
                if text:
                    logger.debug(f"Recognized text: '{text}'")
                    handle_recognized_text(text)
        except Exception:
            break
    return audio_processed

def handle_recognized_text(text: str) -> None:
    """Handle recognized text and trigger processing if wake word detected"""
    text_lower = text.lower()
    if config.WAKE_WORD in text_lower and not state.processing_event.is_set():
        logger.info(f"Wake word '{config.WAKE_WORD}' detected!")
        with state.frame_lock:
            frame_snapshot = state.latest_frame.copy() if state.latest_frame is not None else None
        if frame_snapshot is not None:
            logger.info("Starting command processing thread...")
            threading.Thread(
                target=command_processor.process_command,
                args=(frame_snapshot,),
                daemon=True,
                name="ProcessingThread"
            ).start()
        else:
            logger.warning("No frame available for processing")
    elif config.WAKE_WORD in text_lower:
        logger.info("Wake word detected but processing already in progress")

if __name__ == "__main__":
    main()
