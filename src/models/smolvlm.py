from typing import Any
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

DEFAULT_CKPT = "HuggingFaceTB/SmolVLM-Instruct"

class SmolVLMRunner:
    def __init__(self, checkpoint: str = DEFAULT_CKPT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageTextToText.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto" if self.device=="cuda" else None)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def generate(self, image: Image.Image, task: str = "caption", prompt: str = None, max_new_tokens: int = 64) -> str:
        text = prompt or "Describe this image."
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]