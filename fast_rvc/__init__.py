from .core import RVCInference
from .config import Config

def __getattr__(name):
    if name == "AudioCallbacks":
        from .realtime.callbacks import AudioCallbacks
        return AudioCallbacks
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = ["RVCInference", "Config", "AudioCallbacks"]
