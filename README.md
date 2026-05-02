# Fast RVC Inference

A distilled, standalone library for Retrieval-Based Voice Conversion (RVC), optimized for Apple Silicon (MPS) and in-process usage.

## Features
- **In-process Inference:** No networking overhead.
- **Numpy Array Support:** Direct processing of audio data in memory.
- **Model Caching:** Models stay in memory for fast subsequent calls.
- **Apple Silicon Optimized:** Native support for MPS (Metal Performance Shaders).

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
# Base models (HuBERT, RMVPE, etc.) will be automatically downloaded 
# and cached in ~/.cache/fast-rvc-inference if not found.
rvc = RVCInference()

# 2. Load your specific voice model
rvc.load_model(
    model_path="path/to/MyVoice.pth",
    index_path="path/to/MyVoice.index"
)
```

# 3. Infer from numpy array
audio, sr = librosa.load("input.wav", sr=16000)
out_audio, out_sr = rvc.infer(
    audio_data=audio,
    sr=sr,
    pitch=0,
    f0_method="rmvpe"
)

# 4. Save result
sf.write("output.wav", out_audio, out_sr)
```

## Hardware/Architecture Considerations
- This library is tested on M4 Pro (Apple Silicon).
- It sets `PYTORCH_ENABLE_MPS_FALLBACK=1` to ensure compatibility.
- Ensure you have `faiss-cpu` or `faiss` installed correctly to avoid segmentation faults.
