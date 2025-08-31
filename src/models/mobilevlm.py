from typing import Any
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

DEFAULT_CKPT = "mtgv/MobileVLM_V2-7B"  # small variants also exist

class MobileVLMRunner:
    def __init__(self, checkpoint: str = DEFAULT_CKPT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto" if self.device=="cuda" else None, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    def generate(self, image: Image.Image, task: str = "caption", prompt: str = None, max_new_tokens: int = 64) -> str:
        q = prompt or "Describe this image."
        inputs = self.processor(text=q, images=image, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]