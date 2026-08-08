"""
MiDaS to ONNX Export Script
Downloads pre-exported ONNX models or exports compatible models
"""

import torch
import argparse
import os
import urllib.request
from pathlib import Path


# Pre-exported ONNX models from official sources
ONNX_MODEL_URLS = {
    "midas_v21_small": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
    "midas_v21": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.onnx",
}


def download_progress(count, block_size, total_size):
    """Show download progress"""
    percent = int(count * block_size * 100 / total_size)
    print(f"\rDownloading: {percent}%", end="", flush=True)


def download_onnx_model(model_name: str, output_path: str) -> str:
    """
    Download pre-exported MiDaS ONNX model
    
    Args:
        model_name: Name of the model to download
        output_path: Path to save the model
    """
    if model_name not in ONNX_MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(ONNX_MODEL_URLS.keys())}")
    
    url = ONNX_MODEL_URLS[model_name]
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"Model already exists: {output_path}")
        return output_path
    
    print(f"Downloading {model_name} from {url}")
    urllib.request.urlretrieve(url, output_path, download_progress)
    print(f"\nSaved to: {output_path}")
    
    return output_path


def export_midas_small_onnx(output_path: str, input_size: int = 256):
    """
    Export MiDaS small model to ONNX (this one exports cleanly)
    
    Args:
        output_path: Path to save the ONNX model
        input_size: Input image size
    """
    print("Loading MiDaS small model...")
    
    # Use midas_v21_small which exports cleanly
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.eval()
    
    # Create dummy input with fixed size
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    print(f"Exporting to ONNX: {output_path}")
    
    # Export with fixed input size (no dynamic axes for stability)
    torch.onnx.export(
        midas,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )
    
    print(f"Successfully exported MiDaS small to: {output_path}")
    print(f"Model input size: {input_size}x{input_size}")
    
    # Verify the exported model
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("ONNX model verification passed!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Get MiDaS ONNX model")
    parser.add_argument("--model", type=str, default="midas_v21_small",
                        choices=["midas_v21_small", "midas_v21", "export_small"],
                        help="Model to download or 'export_small' to export locally")
    parser.add_argument("--output", type=str, default="models/midas.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--input-size", type=int, default=256,
                        help="Input size for export (only for export_small)")
    
    args = parser.parse_args()
    
    if args.model == "export_small":
        export_midas_small_onnx(args.output, args.input_size)
    else:
        download_onnx_model(args.model, args.output)
        
    print("\n✅ Done! You can now use the model with video_depth_estimation.py")
    print(f"   python video_depth_estimation.py --video your_video.mp4 --model {args.output}")


if __name__ == "__main__":
    main()
