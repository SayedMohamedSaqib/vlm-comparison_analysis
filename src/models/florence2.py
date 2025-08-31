from typing import Dict, Any
import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from PIL import Image

DEFAULT_CKPT = "microsoft/Florence-2-base"

class Florence2Runner:
    def __init__(self, checkpoint: str = DEFAULT_CKPT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Florence2ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto" if self.device=="cuda" else None)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def generate(self, image: Image.Image, task: str = "caption", prompt: str = None, max_new_tokens: int = 64) -> str:
        if task.lower() in ["caption", "describe"]:
            query = "<CAPTION>"
        else:
            # fallback to freeform
            query = prompt or "Describe the image."
        inputs = self.processor(text=query, images=image, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Post-process for task tags if any
        try:
            out = self.processor.post_process_generation(generated_text, task="<CAPTION>", image_size=(image.width, image.height))
            if isinstance(out, dict) and "caption" in out:
                return out["caption"]
        except Exception:
            pass
        return generated_text