# adas-lite — FCW + BSD stack built with PipeGen

A lightweight L2 driver-assist prototype: **Forward Collision Warning** (front
camera) and **Blind Spot Detection** (rear camera) from plain video, using
TensorRT **FP16** perception generated with
[CraftifAI PipeGen](https://developer.craftifai.com) and a small **C++**
decision layer.

## What it does

- **FCW**: tracks the lead vehicle in the ego corridor, estimates range by
  fusing MiDaS relative depth with a bbox pinhole anchor, computes TTC from the
  closing rate, and raises warn/critical banners on the TTC ladder.
- **BSD**: watches trapezoidal blind-spot zones on the rear view; a vehicle
  that persists in a zone for N frames raises a left/right blind-spot warning.
- Every demo frame carries live **FPS metrics** and detection overlays.

## Architecture

```mermaid
flowchart TB
    F[front camera video] --> P1[PipeGen pipeline - front]
    R[rear camera video] --> P2[PipeGen pipeline - rear]

    subgraph P1F [front perception - TensorRT FP16]
        M2[YOLOv8 - vehicles]
        M3[MiDaS small - relative depth]
    end
    subgraph P2F [rear perception - TensorRT FP16]
        M4[YOLOv8 - vehicles]
        M5[MiDaS small - relative depth]
    end

    P1 --> P1F --> J1[annotated video + results_front.json]
    P2 --> P2F --> J2[annotated video + results_rear.json]

    J1 --> A[adas_fusion - C++]
    J2 --> A
    A --> FCW[FCW - TTC ladder]
    A --> BSD[BSD - zone persistence]
    FCW & BSD --> D[demo videos with warnings + FPS overlay]
```

## Models (all TensorRT FP16, chosen for speed)

| Task | Model | Engine | Source |
|---|---|---|---|
| Vehicle detection (front + rear) | YOLOv8 (ONNX) | `yolov8m.fp16.engine`, 55 MB | PipeGen benchmark catalog |
| Monocular depth (front + rear) | MiDaS v2.1 small | `midas_small.fp16.engine`, 36 MB | PipeGen benchmark catalog |

Compiled on an RTX 5060 Laptop (Blackwell, sm_120) with TensorRT 10.8.

## Decision layer

| Function | Trigger | Thresholds |
|---|---|---|
| FCW | TTC to lead below ladder | warn 3.2 s, critical 2.3 s, range ≤ 40 m, cooldown 2 s |
| BSD | vehicle persists in blind-spot zone | critical 2.5×5 m / overtake 8×5 m equivalents, 10-frame persistence, leaky decay |

Range estimation: per frame, `scale = median(pinhole_i × midas_i)` over
detected vehicles, then `range_i = scale / midas_i` — MiDaS supplies relative
structure, the bbox pinhole supplies absolute scale.

## The exact prompt given to PipeGen

> I am building a lightweight L2 ADAS perception stack for a dual-camera
> vehicle setup, running real-time on edge hardware. Inputs: two video files,
> front camera front.mov and rear camera rear.mov, no physical cameras. Tasks:
> vehicle detection and monocular depth estimation on both cameras as
> independent parallel tasks. Use the lightest and fastest benchmark models
> available: the compact YOLOv8 ONNX repo for vehicles and MiDaS small for
> depth. Compile every engine in FP16. Outputs: no live display window — write
> the processed annotated video to a file, and also write every detection to
> JSON with per-frame objects including class, confidence, bounding box,
> per-object depth value, so a downstream module can compute FCW and BSD.
> Any scene condition is possible including night and rain.

PipeGen compiling the models to **FP16 TensorRT** engines:

![PipeGen FP16 model compilation](assets/pipegen_fp16_pipeline.png)

The full PipeGen session — requirement conversation, repo/source selection,
PIR, and the FP16 model-compilation reports — is included verbatim in
[`pipegen_session/`](pipegen_session/): `RequirementState.json`, `PipelineIR.json`,
`RepoGate.json`, `state.json`, and the per-model reports under `model_stage/`.

## What was built on top of PipeGen

- `adas/` — **adas_fusion** (C++17, OpenCV + nlohmann-json): FCW TTC + BSD
  zone logic, monocular range fusion, warning banners, FPS overlay rendering.
- `pipelines/perception.py` — TensorRT FP16 runner (polygraphy) mirroring the
  PipeGen pipeline output schema, used to iterate while the PipeGen session
  compiled; produces identical `results.json` + annotated video.
- Blackwell (sm_120) container fix: TensorRT 10.3 → 10.8 upgrade of the
  DeepStream 7.1 image (stock image fails with `Unsupported SM: 0xc00`).

## Demos

All demo videos are saved in the [`demos/`](demos/) directory:

| File | Shows |
|---|---|
| [`demos/live_sidebyside.mp4`](demos/live_sidebyside.mp4) | **front FCW and rear BSD side by side** — the main demo |
| [`demos/demo_front_fcw.mp4`](demos/demo_front_fcw.mp4) | FCW on dashcam front view — lead tracking + TTC + range |
| [`demos/demo_rear_bsd.mp4`](demos/demo_rear_bsd.mp4) | BSD blind-spot zones on rear view |

Reproduce the side-by-side demo with one command:

```bash
./run_demo.sh                            # uses the bundled front/rear clips
./run_demo.sh my_front.mp4 my_rear.mp4   # or your own
```

## Build & run

All paths below are relative to the repo root. `run_demo.sh` does all of this
for you; the manual steps are:

```bash
# 0. build the C++ decision layer (once)
cmake -S adas -B adas/build && cmake --build adas/build

# 1. perception — runs PipeGen's FP16 engines (YOLOv8 + MiDaS) in the container,
#    writing an annotated video + per-frame JSON. Run from the repo root.
docker exec craftifai-amd64-v3 bash -c 'cd '"$PWD"'/pipelines && \
  python3 perception.py ~/personal/pipegen/front.mov out_pipegen/results_front.mp4 \
    out_pipegen/results_front.json \
    engines_pipegen/yolov8m.fp16.engine engines_pipegen/midas_small.fp16.engine'

# 2. decision layer + demo render (host)
./adas/build/adas_fusion --mode front --video ~/personal/pipegen/front.mov \
    --json pipelines/out_pipegen/results_front.json --out demos/demo_front_fcw.mp4
./adas/build/adas_fusion --mode rear  --video ~/personal/pipegen/rear.mov \
    --json pipelines/out_pipegen/results_rear.json  --out demos/demo_rear_bsd.mp4
```
