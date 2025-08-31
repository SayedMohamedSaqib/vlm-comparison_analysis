import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

DEFAULT_CKPT = "google/paligemma2-3b-mix-224"

class PaliGemmaRunner:
    def __init__(self, checkpoint: str = DEFAULT_CKPT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto" if self.device=="cuda" else None)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def generate(self, image: Image.Image, task: str = "caption", prompt: str = None, max_new_tokens: int = 64) -> str:
        q = prompt or "What is in this image?"
        inputs = self.processor(image, q, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(out[0], skip_special_tokens=True)