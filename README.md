# VLM Edge/CPU Benchmark – Capstone

A reproducible, side‑by‑side evaluation of small/efficient Vision‑Language Models on a laptop (CPU/GPU) with an optional Raspberry Pi 5 simulation later.

**Models**: Microsoft Florence‑2 (base), SmolVLM (1.7B), MobileVLM‑V2 (1.7B/3B), BLIP‑2 (OPT‑2.7B or FLAN‑T5‑XL), CLIP+Small LLM (toy pipeline), Qwen2‑VL (2B or 7B), PaliGemma (3B‑mix).

**Metrics** (first pass):
- Latency per image (end‑to‑end) and tokens/sec (if available)
- Peak RAM and peak VRAM (if CUDA available)
- Output length (tokens/chars)
- **CLIPScore** (cosine similarity between image and generated caption using CLIP embeds) as a light proxy for caption relevance

> This repo only contains *our code* under MIT. Individual models have their own licenses; accept and comply with them when you `from_pretrained()`.

## Quickstart

```bash
# 0) Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 1) Install deps
pip install -r requirements.txt

# 2) (Optional) Speed-ups if you have a GPU
pip install flash-attn --no-build-isolation  # if supported by your setup
```

### Run a single model on a single image
```bash
python src/bench.py --model florence2 --images data/sample_images/cat.jpg --task caption
```

### Run all models on a folder of images
```bash
python src/bench.py --model all --images data/sample_images --recursive --task caption --out reports/bench.csv
```

### Compare results
- A CSV is saved with timing + memory + CLIPScore.
- You can plot results later in a notebook at `notebooks/analysis.ipynb`.

## Raspberry Pi 5 (later)
We will emulate on x86 first (Docker + ARM64), then test on real Pi 5. Instructions will be added in a separate doc.