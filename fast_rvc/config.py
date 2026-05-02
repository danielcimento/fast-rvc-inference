import torch
import json
import os

class Config:
    def __init__(self, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.is_half = True if self.device != "cpu" else False
        
        # Default RVC config values
        self.x_pad, self.x_query, self.x_center, self.x_max = (1, 6, 38, 41)
        
        if self.device == "mps":
            # MPS and libomp specific fixes for macOS stability
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            # Prevent OpenMP conflicts which cause segfaults on Mac
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            os.environ["OMP_NUM_THREADS"] = "1"

    def __repr__(self):
        return f"Config(device='{self.device}', is_half={self.is_half})"
