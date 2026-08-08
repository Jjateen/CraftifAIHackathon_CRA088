#!/usr/bin/env bash
# run_demo.sh — full ADAS pipeline, front + rear side by side, for screen recording.
#
#   ./run_demo.sh [front_video] [rear_video]
#
# Runs PipeGen's FP16 TensorRT engines (YOLOv8 + MiDaS) on both videos inside
# the PipeGen container, fuses depth+detection into FCW (front) and BSD (rear)
# warnings with adas_fusion, stitches the two annotated streams side by side,
# and plays the result fullscreen. Everything is FP16 end to end.
set -euo pipefail
cd "$(dirname "$0")"
A="$(pwd)"
FRONT="${1:-$HOME/personal/pipegen/front.mov}"
REAR="${2:-$HOME/personal/pipegen/rear.mov}"
CT=craftifai-amd64-v3
YE=pipelines/engines_pipegen/yolov8m.fp16.engine
ME=pipelines/engines_pipegen/midas_small.fp16.engine
mkdir -p pipelines/out_pipegen demos

echo "[1/4] perception (YOLOv8 + MiDaS, FP16 TensorRT) — front"
docker exec "$CT" bash -c "cd $A/pipelines && python3 perception.py '$FRONT' \
  out_pipegen/live_front.mp4 out_pipegen/live_front.json $YE $ME" 2>/dev/null
echo "[2/4] perception — rear"
docker exec "$CT" bash -c "cd $A/pipelines && python3 perception.py '$REAR' \
  out_pipegen/live_rear.mp4 out_pipegen/live_rear.json $YE $ME" 2>/dev/null

echo "[3/4] FCW (front) + BSD (rear) fusion"
./adas/build/adas_fusion --mode front --video "$FRONT" \
  --json pipelines/out_pipegen/live_front.json --out demos/live_front.mp4 >/dev/null
./adas/build/adas_fusion --mode rear  --video "$REAR" \
  --json pipelines/out_pipegen/live_rear.json  --out demos/live_rear.mp4 >/dev/null

echo "[4/4] stitch side by side -> demos/live_sidebyside.mp4"
# scale both to 960 wide, stack horizontally, label each half
ffmpeg -y -v error \
  -i demos/live_front.mp4 -i demos/live_rear.mp4 -filter_complex \
  "[0:v]scale=960:-2,drawtext=text='FRONT — FCW':x=20:y=20:fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.5[l]; \
   [1:v]scale=960:-2,drawtext=text='REAR — BSD':x=20:y=20:fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.5[r]; \
   [l][r]hstack=inputs=2[v]" -map "[v]" demos/live_sidebyside.mp4

echo "playing demos/live_sidebyside.mp4 — press q to quit"
ffplay -v error -autoexit -fs demos/live_sidebyside.mp4
