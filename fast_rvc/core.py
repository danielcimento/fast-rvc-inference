import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Union, Tuple

# Add the current directory to sys.path to handle internal imports
now_dir = os.path.dirname(os.path.abspath(__file__))

from .converter import VoiceConverter
from .config import Config

class RVCInference:
    def __init__(self, device: Optional[str] = None, models_dir: Optional[str] = None):
        """
        Initialize the RVC inference engine.
        
        Args:
            device: 'cpu', 'cuda', or 'mps'. If None, it will be automatically detected.
            models_dir: Directory where models (embedders, predictors) are stored. 
                       Defaults to ~/.cache/fast-rvc-inference.
        """
        self.config = Config(device=device)
        self.models_dir = models_dir or os.path.join(os.path.expanduser("~"), ".cache", "fast-rvc-inference")
        self.converter = VoiceConverter(self.config, models_dir=self.models_dir)
        self.loaded_model_path = None
        self.loaded_index_path = None

    def load_model(self, model_path: str, index_path: Optional[str] = None):
        """
        Load a model and index into memory.
        
        Args:
            model_path: Path to the .pth model file.
            index_path: Optional path to the .index file.
        """
        self.converter.load_model(model_path)
        if index_path:
            self.loaded_index_path = index_path
        self.loaded_model_path = model_path

    def infer(
        self,
        audio_data: Union[np.ndarray, str],
        sr: Optional[int] = None,
        pitch: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        volume_envelope: float = 1.0,
        protect: float = 0.33,
        hop_length: int = 128,
        filter_radius: int = 3,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        embedder_model: str = "contentvec",
        resample_sr: int = 0,
    ) -> Tuple[np.ndarray, int]:
        """
        Perform voice conversion on audio data.
        
        Args:
            audio_data: Numpy array of audio or path to audio file.
            sr: Sample rate of the audio_data (if numpy array).
            pitch: Pitch shift in semitones.
            f0_method: 'rmvpe', 'fcpe', 'crepe', etc.
            index_rate: Rate of index influence (0 to 1).
            volume_envelope: Volume envelope rate.
            protect: Protection rate (0 to 0.5).
            hop_length: Hop length for F0 extraction.
            f0_autotune: Whether to use autotune.
            f0_autotune_strength: Strength of autotune.
            embedder_model: Name of the embedder model.
            resample_sr: Target sample rate for output (0 for model default).
            
        Returns:
            Tuple of (output_audio_numpy, output_sample_rate)
        """
        if isinstance(audio_data, str):
            audio, sr = librosa.load(audio_data, sr=None)
        else:
            audio = audio_data
            if sr is None:
                raise ValueError("sr must be provided if audio_data is a numpy array")

        # Normalize audio to 16k for RVC internal processing
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Ensure audio is float32 and mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        audio = audio.astype(np.float32)

        output_audio = self.converter.convert_audio(
            audio=audio,
            model_path=self.loaded_model_path,
            index_path=self.loaded_index_path,
            pitch=pitch,
            f0_method=f0_method,
            index_rate=index_rate,
            volume_envelope=volume_envelope,
            protect=protect,
            hop_length=hop_length,
            filter_radius=filter_radius,
            f0_autotune=f0_autotune,
            f0_autotune_strength=f0_autotune_strength,
            embedder_model=embedder_model,
            resample_sr=resample_sr,
        )
        
        return output_audio, self.converter.tgt_sr
