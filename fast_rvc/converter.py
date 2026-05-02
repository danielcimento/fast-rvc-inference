import os
import sys
import torch
import numpy as np
import logging
from .pipeline import Pipeline
from .utils import load_embedding
from .algorithm.synthesizers import Synthesizer
from .split_audio import process_audio, merge_audio

class VoiceConverter:
    def __init__(self, config, models_dir=None):
        self.config = config
        self.models_dir = models_dir or os.getcwd()
        self.hubert_model = None
        self.last_embedder_model = None
        self.tgt_sr = None
        self.net_g = None
        self.vc = None
        self.cpt = None
        self.version = None
        self.n_spk = None
        self.use_f0 = None
        self.loaded_model = None

    def load_hubert(self, embedder_model: str, embedder_model_custom: str = None):
        self.hubert_model = load_embedding(embedder_model, embedder_model_custom, models_root=self.models_dir)
        self.hubert_model = self.hubert_model.to(self.config.device)
        if self.config.is_half:
            self.hubert_model = self.hubert_model.half()
        else:
            self.hubert_model = self.hubert_model.float()
        self.hubert_model.eval()

    def load_model(self, model_path: str):
        if self.loaded_model == model_path:
            return
        
        self.cpt = torch.load(model_path, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
        self.use_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")
        self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
        self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")
        
        self.net_g = Synthesizer(
            *self.cpt["config"],
            use_f0=self.use_f0,
            text_enc_hidden_dim=self.text_enc_hidden_dim,
            vocoder=self.vocoder,
        )
        del self.net_g.enc_q
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g = self.net_g.to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()
        self.net_g.eval()
        
        self.vc = Pipeline(self.tgt_sr, self.config, models_dir=self.models_dir)
        self.loaded_model = model_path

    def convert_audio(
        self,
        audio: np.ndarray,
        model_path: str,
        index_path: str = None,
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
        split_audio: bool = False,
        **kwargs,
    ):
        if self.loaded_model != model_path:
            self.load_model(model_path)

        if not self.hubert_model or embedder_model != self.last_embedder_model:
            self.load_hubert(embedder_model)
            self.last_embedder_model = embedder_model

        if resample_sr >= 16000:
            self.tgt_sr = resample_sr

        if split_audio:
            chunks, intervals = process_audio(audio, 16000)
        else:
            chunks = [audio]
            intervals = None

        converted_chunks = []
        for c in chunks:
            audio_opt = self.vc.pipeline(
                model=self.hubert_model,
                net_g=self.net_g,
                sid=0,
                audio=c,
                pitch=pitch,
                f0_method=f0_method,
                file_index=index_path if index_path else "",
                index_rate=index_rate,
                pitch_guidance=self.use_f0,
                volume_envelope=volume_envelope,
                version=self.version,
                protect=protect,
                filter_radius=filter_radius,
                f0_autotune=f0_autotune,
                f0_autotune_strength=f0_autotune_strength,
                proposed_pitch=False,
                proposed_pitch_threshold=155.0,
            )
            converted_chunks.append(audio_opt)

        if split_audio:
            audio_opt = merge_audio(chunks, converted_chunks, intervals, 16000, self.tgt_sr)
        else:
            audio_opt = converted_chunks[0]
        
        return audio_opt
