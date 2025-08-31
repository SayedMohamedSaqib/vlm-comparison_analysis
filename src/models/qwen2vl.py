from typing import Any
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

DEFAULT_CKPT = "Qwen/Qwen2-VL-2B-Instruct"

class Qwen2VLRunner:
    def __init__(self, checkpoint: str = DEFAULT_CKPT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto" if self.device=="cuda" else None)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def generate(self, image: Image.Image, task: str = "caption", prompt: str = None, max_new_tokens: int = 64) -> str:
        q = prompt or "Describe this image."
        messages = [ { "role":"user", "content":[ {"type":"image"}, {"type":"text", "text": q } ] } ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen = [ o[len(i):] for i,o in zip(inputs.input_ids, out) ]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]