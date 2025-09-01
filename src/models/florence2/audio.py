# audio.py - Updated to find Vosk model in parent directory

import os
import sounddevice as sd
import numpy as np
import time
import traceback
from vosk import Model, KaldiRecognizer
import config
import state

def audio_callback(indata, frames, time_info, status):
    if status:
        print("[audio] status:", status)
    try:
        state.audio_queue.put(bytes(indata))
    except Exception:
        traceback.print_exc()

def start_audio_stream():
    try:
        stream = sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        )
        stream.start()
        return stream
    except Exception as e:
        raise RuntimeError(f"Failed to start audio stream: {e}")

def init_vosk():
    # First try the config path
    vosk_dir = config.VOSK_MODEL_DIR
    
    # If config path doesn't exist, try parent directory
    if not os.path.isdir(vosk_dir):
        # Get the directory where this audio.py file is located (florence2 folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to parent directory
        parent_dir = os.path.dirname(current_dir)
        # Look for vosk model in parent directory
        parent_vosk_path = os.path.join(parent_dir, "vosk-model-small-en-us-0.15")
        
        print(f"[audio] Config path not found: {vosk_dir}")
        print(f"[audio] Trying parent directory: {parent_vosk_path}")
        
        if os.path.isdir(parent_vosk_path):
            vosk_dir = parent_vosk_path
            print(f"[audio] Using Vosk model from parent directory: {vosk_dir}")
        else:
            # Also try current directory as fallback
            current_vosk_path = os.path.join(current_dir, "vosk-model-small-en-us-0.15")
            print(f"[audio] Also trying current directory: {current_vosk_path}")
            
            if os.path.isdir(current_vosk_path):
                vosk_dir = current_vosk_path
                print(f"[audio] Using Vosk model from current directory: {vosk_dir}")
            else:
                raise FileNotFoundError(f"Vosk model directory not found in:\n"
                                      f"  Config path: {config.VOSK_MODEL_DIR}\n"
                                      f"  Parent directory: {parent_vosk_path}\n"
                                      f"  Current directory: {current_vosk_path}")
    
    model = Model(vosk_dir)
    recognizer = KaldiRecognizer(model, config.SAMPLE_RATE)
    return recognizer

def record_audio(seconds=7, timeout_sec=9):
    frames_needed = int(seconds * config.SAMPLE_RATE)
    audio_buffer = []
    total_frames = 0
    deadline = time.time() + timeout_sec

    while total_frames < frames_needed:
        remaining = deadline - time.time()
        if remaining <= 0:
            break

        try:
            audio_chunk = state.audio_queue.get(timeout=min(0.1, remaining))
        except Exception:
            continue

        audio_buffer.append(audio_chunk)
        total_frames += len(audio_chunk) // 2  # int16 -> 2 bytes

    # flush any extra
    try:
        while True:
            state.audio_queue.get_nowait()
    except Exception:
        pass

    if not audio_buffer:
        return np.zeros(frames_needed, dtype=np.float32)

    audio_np = np.frombuffer(b"".join(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio_np) < frames_needed:
        pad = np.zeros(frames_needed - len(audio_np), dtype=np.float32)
        audio_np = np.concatenate([audio_np, pad])
    else:
        audio_np = audio_np[:frames_needed]

    return audio_np