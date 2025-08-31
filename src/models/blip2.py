from typing import Any
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image

# Smaller than FLAN-T5-XL variant
DEFAULT_CKPT = "Salesforce/blip2-opt-2.7b"

class BLIP2Runner:
    def __init__(self, checkpoint: str = DEFAULT_CKPT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Blip2ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto" if self.device=="cuda" else None)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def generate(self, image: Image.Image, task: str = "caption", prompt: str = None, max_new_tokens: int = 64) -> str:
        q = prompt or "A photo of"
        inputs = self.processor(images=image, text=q, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()