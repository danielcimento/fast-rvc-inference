import argparse
import soundfile as sf
from fast_rvc.core import RVCInference

def main():
    parser = argparse.ArgumentParser(description="Fast RVC Inference CLI")
    
    # Basic required/core args
    parser.add_argument("-i", "--input", required=True, help="Input audio file path")
    parser.add_argument("-o", "--output", required=True, help="Output audio file path")
    parser.add_argument("-m", "--model", required=True, help="Path to the model (.pth) file")
    parser.add_argument("-x", "--index", default=None, help="Optional path to the index (.index) file")
    
    # RVCInference initialization args
    parser.add_argument("-d", "--device", default=None, help="Device to use ('cpu', 'cuda', 'mps')")
    parser.add_argument("--models-dir", default=None, help="Directory where models are stored")
    
    # Inference args
    parser.add_argument("-p", "--pitch", type=int, default=0, help="Pitch shift in semitones")
    parser.add_argument("-f", "--f0-method", default="rmvpe", help="F0 extraction method ('rmvpe', 'fcpe', 'crepe', etc.)")
    parser.add_argument("-r", "--index-rate", type=float, default=0.75, help="Rate of index influence (0 to 1)")
    parser.add_argument("-e", "--volume-envelope", type=float, default=1.0, help="Volume envelope rate")
    parser.add_argument("-pr", "--protect", type=float, default=0.33, help="Protection rate (0 to 0.5)")
    parser.add_argument("-hl", "--hop-length", type=int, default=128, help="Hop length for F0 extraction")
    parser.add_argument("-fr", "--filter-radius", type=int, default=3, help="Filter radius")
    parser.add_argument("-a", "--f0-autotune", action="store_true", help="Whether to use autotune")
    parser.add_argument("-as", "--f0-autotune-strength", type=float, default=1.0, help="Strength of autotune")
    parser.add_argument("-em", "--embedder-model", default="contentvec", help="Name of the embedder model")
    parser.add_argument("-rs", "--resample-sr", type=int, default=0, help="Target sample rate for output (0 for model default)")

    args = parser.parse_args()

    print(f"Initializing RVC Inference on {args.device or 'auto'} device...")
    rvc = RVCInference(device=args.device, models_dir=args.models_dir)
    
    print(f"Loading model: {args.model}")
    rvc.load_model(args.model, args.index)
    
    print(f"Running inference on {args.input}...")
    output_audio, output_sr = rvc.infer(
        audio_data=args.input,
        pitch=args.pitch,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        volume_envelope=args.volume_envelope,
        protect=args.protect,
        hop_length=args.hop_length,
        filter_radius=args.filter_radius,
        f0_autotune=args.f0_autotune,
        f0_autotune_strength=args.f0_autotune_strength,
        embedder_model=args.embedder_model,
        resample_sr=args.resample_sr
    )
    
    print(f"Saving output to {args.output}...")
    sf.write(args.output, output_audio, output_sr)
    print("Done!")

if __name__ == "__main__":
    main()
