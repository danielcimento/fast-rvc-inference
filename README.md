# Fast RVC Inference

A distilled, standalone library for Retrieval-Based Voice Conversion (RVC), optimized for Apple Silicon (MPS) and in-process usage.

## Features
- **In-process Inference:** No networking overhead.
- **Numpy Array Support:** Direct processing of audio data in memory.
- **Model Caching:** Models stay in memory for fast subsequent calls.
- **Apple Silicon Optimized:** Native support for MPS (Metal Performance Shaders).
- **Autonomous Model Management:** Automatically downloads and caches base models (HuBERT, RMVPE, FCPE).

## Installation

### Prerequisites
On macOS, it is recommended to install `faiss` via Homebrew for best performance:
```bash
brew install faiss
```

### Install the library
```bash
cd fast-rvc-inference
pip install -e .
```

## Usage

```python
from fast_rvc import RVCInference
import librosa
import soundfile as sf

# 1. Initialize (automatically detects MPS/CUDA/CPU)
# Base models will be cached in ~/.cache/fast-rvc-inference
rvc = RVCInference()

# 2. Load a voice model
rvc.load_model(
    model_path="path/to/voice.pth",
    index_path="path/to/voice.index"
)

# 3. Infer from numpy array
audio, sr = librosa.load("input.wav", sr=16000)
out_audio, out_sr = rvc.infer(
    audio_data=audio,
    sr=sr,
    pitch=0,
    f0_method="rmvpe",
    index_rate=0.75,
    volume_envelope=1.0,
    protect=0.33,
    filter_radius=3
)

# 4. Save result
sf.write("output.wav", out_audio, out_sr)
```

## Supported F0 Methods

| Method | Recommended Hardware | Description |
| :--- | :--- | :--- |
| **RMVPE** | **All (Default)** | Most robust and high-quality method. Native support on Apple Silicon. |
| **FCPE** | **All** | Fast and accurate. Excellent for real-time and batch processing. |
| **CREPE** | NVIDIA GPU | High quality but very slow on CPU/MPS. Requires significant VRAM. |
| **CREPE-TINY**| All | Faster version of CREPE with lower quality. |

**Hardware Note:** RMVPE and FCPE are the best performers on **Apple Silicon (M1/M2/M3/M4)**. They provide the best balance of speed and pitch accuracy without the overhead of CREPE.

## Advanced Parameters
- `index_rate`: (0.0 - 1.0) Controls how much of the voice's accent/style is retrieved from the `.index` file.
- `volume_envelope`: (0.0 - 1.0) Blends the volume envelope of the input audio with the output.
- `protect`: (0.0 - 0.5) Protects voiceless consonants and breath sounds from being "pitched."
- `filter_radius`: (0 - 7) Median filtering radius applied to the pitch contour to reduce artifacts.

## Hardware/Architecture Considerations
- This library is tested on M4 Pro (Apple Silicon).
- It sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and `KMP_DUPLICATE_LIB_OK=TRUE` for stability.
- Threading is limited to `OMP_NUM_THREADS=1` to prevent initialization crashes on macOS.
