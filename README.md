# Florence-2 Real-Time Vision-Language System

A comprehensive real-time vision-language processing system featuring Microsoft Florence-2 for multimodal understanding, integrated with speech recognition, object tracking, and interactive voice commands for edge deployment scenarios. We will primarily test lightweight models deployable on Edge devices and run a comparitive analysis of each. The repository will continuously be updated as we go through each of the models. As per the latest update we have a completed a pipelin on the Florence2 model, the most promising one yet.

## Features

- **Real-time Vision-Language Processing**: Live webcam feed analysis with Florence-2 model
- **Voice-Controlled Interface**: Wake-word detection with semantic command mapping
- **Multi-Modal Pipeline**: Audio (Vosk + Whisper) → Speech-to-Text → Task Mapping → Vision Processing
- **Object Tracking**: ByteTrack integration for persistent object identification
- **Edge-Ready**: Optimized for CPU/GPU deployment with configurable performance settings
- **Extensible Architecture**: Modular design supporting additional VLM models

## Quick Start

### Prerequisites

- Python 3.10+
- Webcam and microphone access
- Optional: CUDA-enabled GPU for acceleration

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional GPU acceleration (if supported)
pip install flash-attn --no-build-isolation
```

### Download Vosk Model

```bash
# Download and extract Vosk speech recognition model
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

### Run the Application

```bash
python main.py
```

## Usage

### Interactive Mode

1. **Start the application**: `python main.py`
2. **Wait for initialization**: Models will load automatically
3. **Activate with wake word**: Say "wake up" (configurable in `config.py`)
4. **Give voice commands**: Speak natural language commands like:
   - "describe the image"
   - "detect objects"
   - "find text in the image"
   - "segment objects"

### Supported Tasks

The system supports multiple Florence-2 vision tasks triggered by semantic voice commands:

| Task Category        | Voice Commands                                             | Florence-2 Task      |
| -------------------- | ---------------------------------------------------------- | -------------------- |
| **Object Detection** | "detect objects", "find items", "what objects are present" | `<OD>`               |
| **Image Captioning** | "describe image", "what do you see", "caption this"        | `<CAPTION>`          |
| **Dense Captioning** | "detailed description", "comprehensive caption"            | `<DETAILED_CAPTION>` |
| **OCR**              | "read text", "extract words", "what does it say"           | `<OCR>`              |
| **Segmentation**     | "segment objects", "create masks"                          | `<REGION_PROPOSAL>`  |

### Configuration

Key settings in `config.py`:

```python
# Audio Configuration
WAKE_WORD = "wake up"           # Trigger phrase
SAMPLE_RATE = 16000             # Audio sampling rate

# Processing Settings
MIN_PROCESS_INTERVAL = 0.5      # Throttling between commands
SEMANTIC_MATCH_THRESHOLD = 0.6  # Speech-to-task matching sensitivity

# Model Settings
PROC_SIZE_DEFAULT = (224, 224)  # Default processing resolution
PROC_SIZE_REFERRING = (480, 480) # High-res for complex tasks
```

## Architecture

### Core Components

- **`main.py`**: Application entry point and orchestration
- **`models.py`**: Model loading and management (Florence-2, Whisper, SentenceTransformers)
- **`audio.py`**: Audio capture and Vosk wake-word detection
- **`command_processor.py`**: Speech-to-vision pipeline coordination
- **`processor.py`**: Semantic task mapping from speech to Florence-2 tasks
- **`video.py`**: Webcam capture, tracking, and visualization
- **`tracker.py`**: ByteTracker configuration and object tracking
- **`utils.py`**: Bounding box parsing and image processing utilities
- **`state.py`**: Thread-safe global state management

### Data Flow

```
Microphone → Vosk (Wake Word) → Audio Recording → Whisper (STT)
     ↓
Semantic Mapping → Florence-2 Task Selection → Image Processing
     ↓
Webcam Frame → Florence-2 Model → Results (Captions/Detections/Masks)
     ↓
ByteTracker → Visual Overlay → Display
```

## 🔧 Technical Details

### Models Used

- **Florence-2 Base**: `microsoft/Florence-2-base` - Primary vision-language model
- **Whisper Small**: `openai/whisper-small` - Speech transcription
- **Sentence Transformers**: `all-MiniLM-L6-v2` - Semantic similarity for task mapping
- **Vosk**: `vosk-model-small-en-us-0.15` - Offline wake-word detection
- **ByteTracker**: Object tracking across video frames

### Performance Considerations

- **CPU Mode**: Fallback support with float32 precision
- **GPU Mode**: CUDA acceleration with float16 precision when available
- **Memory Management**: Automatic model cleanup and CUDA cache clearing
- **Processing Throttling**: Configurable intervals to prevent system overload

### Thread Safety

The system uses multiple threads for concurrent processing:

- Main thread: Audio processing and wake-word detection
- Video thread: Webcam capture and display
- Processing thread: Vision-language model inference
- All shared state protected by appropriate locks

## 📊 Benchmarking & Evaluation

### Metrics Tracked

- **Latency**: End-to-end processing time (audio → vision → results)
- **Memory Usage**: Peak RAM and VRAM consumption
- **FPS**: Real-time video processing performance
- **Detection Quality**: Bounding box accuracy and tracking consistency
- **Caption Relevance**: Semantic similarity between generated text and image content

### Performance Monitoring

Access real-time statistics via the state management system:

```python
from state import get_stats, get_state_summary

# Get processing statistics
stats = get_stats()
print(f"Avg processing time: {stats['processing_time_avg']:.3f}s")

# Get system state summary
summary = get_state_summary()
print(f"Objects detected: {summary['detection_count']}")
```

## 🎛️ Customization

### Adding New Tasks

1. **Define task prompts** in `processor.py`:

```python
task_prompts["<NEW_TASK>"] = [
    "custom command 1",
    "custom command 2"
]
```

2. **Handle task output** in `command_processor.py`:

```python
elif task_code == "<NEW_TASK>":
    # Custom processing logic
    custom_result = process_custom_task(parsed_answer)
```

### Extending Model Support

The modular architecture supports adding new VLM models:

1. Add model loading in `models.py`
2. Implement processing logic in `command_processor.py`
3. Update task mapping in `processor.py`

## 🚀 Future Enhancements

### Planned Features

- **Multi-VLM Support**: Integration of additional models (SmolVLM, PaliGemma, Qwen2-VL)
- **Raspberry Pi 5 Deployment**: ARM64 optimization and edge deployment
- **Batch Processing**: CLI interface for non-interactive benchmarking
- **Model Quantization**: INT8/INT4 optimization for resource-constrained environments
- **Advanced Tracking**: Multi-object persistent identity across sessions

### Raspberry Pi 5 Roadmap

1. **Docker Emulation**: x86 → ARM64 cross-compilation testing
2. **Performance Profiling**: Memory and compute optimization for Pi hardware
3. **Model Compression**: Quantization and pruning for 8GB RAM constraint
4. **Real-Device Validation**: End-to-end testing on physical Raspberry Pi 5

## 📝 Requirements

See `requirements.txt` for complete dependency list. Key packages:

- `torch>=2.3` - Deep learning framework
- `transformers>=4.45` - Hugging Face model library
- `ultralytics>=8.0.20` - Object tracking
- `opencv-python>=4.9.0` - Computer vision
- `sounddevice==0.5.2` - Audio capture
- `vosk>=0.3.45` - Speech recognition
- `sentence-transformers>=2.2.2` - Semantic similarity

## 📄 License

This project code is released under the MIT License. Individual pre-trained models retain their original licenses:

- Florence-2: Microsoft Research License
- Whisper: MIT License
- Sentence Transformers: Apache 2.0
- Vosk: Apache 2.0
- Ultralytics: AGPL 3.0

## 🤝 Contributing

Contributions welcome! Please see our contributing guidelines for:

- Code style and formatting standards
- Testing requirements
- Documentation expectations
- Performance benchmarking protocols

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**:

- Ensure sufficient RAM/VRAM (recommend 8GB+ system RAM, 4GB+ VRAM)
- Check internet connectivity for initial model downloads

**Audio Issues**:

- Verify microphone permissions and device access
- Ensure Vosk model is properly downloaded and extracted
- Check audio device compatibility with `sounddevice`

**Video Issues**:

- Confirm webcam availability and permissions
- Try different camera indices if default fails: `cv2.VideoCapture(1)`

**Performance Issues**:

- Reduce processing resolution in `config.py`
- Increase `MIN_PROCESS_INTERVAL` for slower systems
- Enable GPU acceleration if available

For detailed logs, set environment variable: `LOG_LEVEL=DEBUG`
