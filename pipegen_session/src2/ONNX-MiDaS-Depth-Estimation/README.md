# ONNX MiDaS Video Depth Estimation

Video depth estimation pipeline using MiDaS ONNX model with ONNX Runtime.

## Features

- 🎥 Real-time video depth estimation
- 🚀 GPU acceleration with ONNX Runtime
- 📹 Support for video files and webcam
- 🎨 Multiple colormap options
- 💾 Save output as video

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Export MiDaS to ONNX

First, export the MiDaS model to ONNX format:

```bash
# Export DPT_Large (best quality)
python export_midas_onnx.py --model-type DPT_Large --output models/midas_dpt_large.onnx

# Export DPT_Hybrid (balanced)
python export_midas_onnx.py --model-type DPT_Hybrid --output models/midas_dpt_hybrid.onnx

# Export MiDaS_small (fastest)
python export_midas_onnx.py --model-type MiDaS_small --output models/midas_small.onnx --input-size 256
```

### Step 2: Run Video Depth Estimation

```bash
# Process a video file
python video_depth_estimation.py --video input.mp4 --model models/midas_dpt_large.onnx --output output.mp4

# Use webcam (camera 0)
python video_depth_estimation.py --video 0 --model models/midas_dpt_large.onnx

# Use CPU only
python video_depth_estimation.py --video input.mp4 --model models/midas_dpt_large.onnx --cpu

# Different colormap
python video_depth_estimation.py --video input.mp4 --model models/midas_dpt_large.onnx --colormap jet
```

## Command Line Options

| Option         | Description                                          | Default        |
| -------------- | ---------------------------------------------------- | -------------- |
| `--video`      | Input video path or camera index                     | Required       |
| `--model`      | Path to MiDaS ONNX model                             | Required       |
| `--output`     | Output video path                                    | None (no save) |
| `--input-size` | Model input size                                     | 384            |
| `--cpu`        | Force CPU execution                                  | False          |
| `--no-display` | Disable live display                                 | False          |
| `--colormap`   | Depth colormap (inferno, jet, turbo, viridis, magma) | inferno        |

## Model Comparison

| Model       | Input Size | Speed  | Quality    |
| ----------- | ---------- | ------ | ---------- |
| DPT_Large   | 384x384    | Slow   | Best       |
| DPT_Hybrid  | 384x384    | Medium | Good       |
| MiDaS_small | 256x256    | Fast   | Acceptable |

## Output

The output video shows side-by-side comparison:
- Left: Original frame
- Right: Depth map (colorized)

## Controls

- Press `q` to quit during playback
