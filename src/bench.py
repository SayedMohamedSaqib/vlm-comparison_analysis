import argparse, os, glob, csv, time
from PIL import Image
from src.common import load_image, PerfMeter
from src.metrics import clipscore

# Import runners
from src.models.florence2 import Florence2Runner
from src.models.smolvlm import SmolVLMRunner
from src.models.mobilevlm import MobileVLMRunner
from src.models.blip2 import BLIP2Runner
from src.models.clip_small_llm import CLIPSmallLLM
from src.models.qwen2vl import Qwen2VLRunner
from src.models.paligemma import PaliGemmaRunner

RUNNERS = {
    "florence2": Florence2Runner,
    "smolvlm": SmolVLMRunner,
    "mobilevlm": MobileVLMRunner,
    "blip2": BLIP2Runner,
    "clip+llm": CLIPSmallLLM,
    "qwen2-vl": Qwen2VLRunner,
    "paligemma": PaliGemmaRunner,
}

def iter_images(path, recursive=False):
    if os.path.isdir(path):
        pattern = "**/*" if recursive else "*"
        for p in glob.glob(os.path.join(path, pattern), recursive=recursive):
            if p.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp','.gif')):
                yield p
    else:
        yield path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="one of: %s, or 'all'" % ",".join(RUNNERS.keys()))
    ap.add_argument("--images", required=True, help="image file or directory")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--task", default="caption")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out", default=None, help="CSV path to save results")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    models = list(RUNNERS.keys()) if args.model == "all" else [args.model]
    rows = []
    for m in models:
        runner = RUNNERS[m]()
        for p in iter_images(args.images, recursive=args.recursive):
            img = load_image(p)
            meter = PerfMeter(); meter.before()
            text = runner.generate(img, task=args.task, prompt=args.prompt, max_new_tokens=args.max_new)
            t, ram, vram = meter.after()
            score = clipscore(img, text)
            rows.append({
                "model": m,
                "image": os.path.basename(p),
                "latency_s": round(t, 3),
                "ram_mb": round(ram, 1),
                "vram_mb_peak": round(vram, 1) if vram is not None else "",
                "clipscore": round(score, 4),
                "output": text.replace("\n"," ").strip()
            })
            print(f"[{m}] {p} -> {text} | {t:.2f}s, RAM {ram:.0f}MB, VRAM {vram if vram else 0:.0f}MB, CLIPScore {score:.3f}")

    fieldnames = ["model","image","latency_s","ram_mb","vram_mb_peak","clipscore","output"]
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()