from setuptools import setup, find_packages

setup(
    name="fast-rvc-inference",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.0",
        "torch>=2.0.0",
        "torchaudio",
        "librosa>=0.10.0",
        "scipy",
        "soundfile",
        "faiss-cpu",
        "transformers>=4.40.0",
        "noisereduce",
        "pedalboard",
        "stftpitchshift",
        "soxr",
        "torchcrepe",
        "torchfcpe",
        "einops",
        "wget",
        "tqdm",
        "webrtcvad-wheels",
    ],
    python_requires=">=3.8",
)
