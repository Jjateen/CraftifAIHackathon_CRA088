"""
Video Depth Estimation using MiDaS ONNX
Processes video frames using ONNX Runtime for depth estimation
"""

import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
from typing import Tuple, Optional


class MiDaSDepthEstimator:
    """MiDaS Depth Estimation using ONNX Runtime"""
    
    def __init__(self, onnx_model_path: str, input_size: int = 384, use_gpu: bool = True):
        """
        Initialize the MiDaS ONNX depth estimator
        
        Args:
            onnx_model_path: Path to the MiDaS ONNX model
            input_size: Model input size (384 for DPT, 256 for small)
            use_gpu: Whether to use GPU acceleration
        """
        self.input_size = input_size
        
        # Setup ONNX Runtime session
        providers = []
        if use_gpu:
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
            }))
        providers.append('CPUExecutionProvider')
        
        print(f"Loading ONNX model: {onnx_model_path}")
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Check which provider is being used
        provider_used = self.session.get_providers()[0]
        print(f"Using provider: {provider_used}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for MiDaS inference
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed input tensor and original image size
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        
        # Normalize (MiDaS uses ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        
        img_normalized = (img_resized.astype(np.float32) / 255.0 - mean) / std
        
        # HWC to CHW, add batch dimension
        input_tensor = img_normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        
        return input_tensor, original_size
    
    def postprocess(self, depth_output: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess depth output
        
        Args:
            depth_output: Raw depth prediction from model
            original_size: Original image size (width, height)
            
        Returns:
            Normalized depth map at original resolution
        """
        depth = depth_output.squeeze()
        
        # Resize to original size
        depth_resized = cv2.resize(depth, original_size)
        
        # Normalize to 0-255 for visualization
        depth_min = depth_resized.min()
        depth_max = depth_resized.max()
        depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)
        
        return depth_normalized
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a single image
        
        Args:
            image: Input BGR image
            
        Returns:
            Normalized depth map (0-255)
        """
        input_tensor, original_size = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        depth_output = outputs[0]
        
        # Postprocess
        depth_map = self.postprocess(depth_output, original_size)
        
        return depth_map


def apply_colormap(depth_map: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Apply colormap to depth map for visualization"""
    return cv2.applyColorMap(depth_map, colormap)


def process_video(
    video_path: str,
    onnx_model_path: str,
    output_path: Optional[str] = None,
    input_size: int = 384,
    use_gpu: bool = True,
    show_display: bool = True,
    colormap: int = cv2.COLORMAP_INFERNO
):
    """
    Process video for depth estimation
    
    Args:
        video_path: Path to input video (or camera index as integer)
        onnx_model_path: Path to MiDaS ONNX model
        output_path: Path to save output video (optional)
        input_size: Model input size
        use_gpu: Whether to use GPU
        show_display: Whether to show live display
        colormap: OpenCV colormap for depth visualization
    """
    # Initialize depth estimator
    estimator = MiDaSDepthEstimator(onnx_model_path, input_size, use_gpu)
    
    # Open video
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    
    # Setup video writer if output path specified
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Output will be side-by-side: original + depth
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    frame_count = 0
    
    print("Processing video... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Estimate depth
        depth_map = estimator.estimate_depth(frame)
        
        # Apply colormap
        depth_colored = apply_colormap(depth_map, colormap)
        
        # Create side-by-side visualization
        combined = np.hstack([frame, depth_colored])
        
        # Add frame info
        cv2.putText(combined, f"Frame: {frame_count}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write to output video
        if writer:
            writer.write(combined)
        
        # Show display
        if show_display:
            cv2.imshow("Depth Estimation (Press 'q' to quit)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
        print(f"Output saved to: {output_path}")
    cv2.destroyAllWindows()
    
    print(f"Processing complete! Total frames: {frame_count}")


def main():
    parser = argparse.ArgumentParser(description="Video Depth Estimation using MiDaS ONNX")
    parser.add_argument("--video", type=str, required=False, default="0",
                        help="Path to input video or camera index (0, 1, etc.)")
    parser.add_argument("--model", type=str, required=False, default="./models/midas_v21_small.onnx", 
                        help="Path to MiDaS ONNX model")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output video (optional)")
    parser.add_argument("--input-size", type=int, default=256,
                        help="Model input size (256 for MiDaS v2.1)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live display")
    parser.add_argument("--colormap", type=str, default="inferno",
                        choices=["inferno", "jet", "turbo", "viridis", "magma"],
                        help="Colormap for depth visualization")
    
    args = parser.parse_args()
    
    # Map colormap names to OpenCV constants
    colormap_map = {
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "magma": cv2.COLORMAP_MAGMA
    }
    
    process_video(
        video_path=args.video,
        onnx_model_path=args.model,
        output_path=args.output,
        input_size=args.input_size,
        use_gpu=not args.cpu,
        show_display=not args.no_display,
        colormap=colormap_map[args.colormap]
    )


if __name__ == "__main__":
    main()
