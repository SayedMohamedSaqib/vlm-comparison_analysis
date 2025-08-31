import os, time, psutil, torch
from PIL import Image

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_image(path_or_url):
    from PIL import Image
    if str(path_or_url).startswith("http"):
        import requests, io
        im = Image.open(io.BytesIO(requests.get(path_or_url, stream=True).content)).convert("RGB")
        return im
    return Image.open(path_or_url).convert("RGB")

class PerfMeter:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.cuda_peak = 0

    def before(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()

    def after(self):
        t = time.time() - self.t0
        ram = self.process.memory_info().rss / (1024**2)
        vram_peak = None
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() / (1024**2)
        return t, ram, vram_peak