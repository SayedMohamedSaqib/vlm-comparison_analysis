import torch, torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from PIL import Image

# CLIPScore proxy: cosine between image embedding and text embedding
_CLIP_NAME = "openai/clip-vit-base-patch32"
_clip_model = None
_clip_proc = None

def _load():
    global _clip_model, _clip_proc
    if _clip_model is None:
        _clip_model = AutoModel.from_pretrained(_CLIP_NAME)
        _clip_model.eval()
    if _clip_proc is None:
        _clip_proc = AutoProcessor.from_pretrained(_CLIP_NAME)

def clipscore(image: Image.Image, text: str) -> float:
    _load()
    with torch.no_grad():
        inputs = _clip_proc(text=[text], images=image, return_tensors="pt", padding=True)
        outputs = _clip_model(**inputs)
        # text/image features before projection are in outputs? For AutoModel(CLIPModel), use get_text_features/get_image_features
        # Fall back to try/except
        try:
            tfeat = outputs.text_embeds
            if tfeat is None:
                raise AttributeError
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                ifeat = outputs.image_embeds
            else:
                raise AttributeError
        except Exception:
            from transformers import CLIPModel
            model = CLIPModel.from_pretrained(_CLIP_NAME)
            model.eval()
            tfeat = model.get_text_features(**_clip_proc(text=[text], return_tensors="pt", padding=True))
            ifeat = model.get_image_features(**_clip_proc(images=image, return_tensors="pt"))
        tfeat = F.normalize(tfeat, dim=-1)
        ifeat = F.normalize(ifeat, dim=-1)
        return float((tfeat @ ifeat.T).squeeze().cpu().item())