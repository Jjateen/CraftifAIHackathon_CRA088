#!/usr/bin/env bash
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")"
yt-dlp -f 'bv*[height<=720][ext=mp4]/bv*[height<=720]/b' --no-playlist -o 'front/front_dashcam.%(ext)s' 'ytsearch1:highway driving dashcam POV 4k'
yt-dlp -f 'bv*[height<=720][ext=mp4]/bv*[height<=720]/b' --no-playlist -o 'carla/carla_demo.%(ext)s' 'ytsearch1:CARLA simulator driving demo front camera'
