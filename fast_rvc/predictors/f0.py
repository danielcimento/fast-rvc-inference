import os
import torch
import wget

from .RMVPE import RMVPE0Predictor
from torchfcpe import spawn_infer_model_from_pt
import torchcrepe
import numpy as np

def download_predictor(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {url} to {path}...")
        wget.download(url, out=path)
        print("\nDownload complete.")

class RMVPE:
    def __init__(self, device, model_name="rmvpe.pt", sample_rate=16000, hop_size=160, models_dir=None):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        if models_dir is None:
            models_dir = os.path.join(os.path.expanduser("~"), ".cache", "fast-rvc-inference")
        
        model_path = os.path.join(models_dir, "rvc", "models", "predictors", model_name)
        if not os.path.exists(model_path):
            url = f"https://huggingface.co/IAHispano/Applio/resolve/main/Resources/predictors/{model_name}"
            download_predictor(url, model_path)
            
        self.model = RMVPE0Predictor(
            model_path,
            device=self.device,
        )

    def get_f0(self, x, filter_radius=0.03):
        f0 = self.model.infer_from_audio(x, thred=filter_radius)
        return f0


class CREPE:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None, model="full"):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        batch_size = 512

        f0, pd = torchcrepe.predict(
            x.float().to(self.device).unsqueeze(dim=0),
            self.sample_rate,
            self.hop_size,
            f0_min,
            f0_max,
            model=model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()

        return f0


class FCPE:
    def __init__(self, device, sample_rate=16000, hop_size=160, models_dir=None):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        if models_dir is None:
            models_dir = os.path.join(os.path.expanduser("~"), ".cache", "fast-rvc-inference")
        
        model_path = os.path.join(models_dir, "rvc", "models", "predictors", "fcpe.pt")
        if not os.path.exists(model_path):
            url = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/predictors/fcpe.pt"
            download_predictor(url, model_path)
            
        self.model = spawn_infer_model_from_pt(
            model_path,
            self.device,
            bundled_model=True,
        )

    def get_f0(self, x, p_len=None, filter_radius=0.006):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        f0 = (
            self.model.infer(
                x.float().to(self.device).unsqueeze(0),
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=filter_radius,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return f0
