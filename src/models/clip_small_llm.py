from typing import List
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, AutoModelForCausalLM, AutoTokenizer
from PIL import Image

CLIP_CKPT = "openai/clip-vit-base-patch32"
LLM_CKPT = "Qwen/Qwen2-0.5B-Instruct"

class CLIPSmallLLM:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = AutoModelForZeroShotImageClassification.from_pretrained(CLIP_CKPT)
        self.clip_processor = AutoProcessor.from_pretrained(CLIP_CKPT)
        self.tok = AutoTokenizer.from_pretrained(LLM_CKPT)
        self.llm = AutoModelForCausalLM.from_pretrained(LLM_CKPT, torch_dtype="auto", device_map="auto" if self.device=="cuda" else None)

    def _tags(self, image: Image.Image, candidates: List[str] = None, k: int = 5) -> List[str]:
        if candidates is None:
            candidates = ["cat","dog","person","car","tree","building","food","computer","phone","book","bottle","chair","table","bicycle","road","sign","flower","animal"]
        inputs = self.clip_processor(images=image, text=[f"This is a photo of {c}." for c in candidates], return_tensors="pt")
        with torch.no_grad():
            logits = self.clip_model(**inputs).logits_per_image[0]
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, min(k, probs.shape[-1]))
        return [candidates[i] for i in topk.indices.tolist()]

    def generate(self, image: Image.Image, task: str = "caption", prompt: str = None, max_new_tokens: int = 64) -> str:
        tags = self._tags(image)
        prompt_text = prompt or f"Write a one-sentence image caption using these tags: {', '.join(tags)}."
        inputs = self.tok(prompt_text, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = self.llm.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tok.decode(out[0], skip_special_tokens=True)