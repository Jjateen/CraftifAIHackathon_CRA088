#!/usr/bin/env bash
# Fetch test clips: real dashcam (front/rear) + CARLA sim renders. ~2 min each, 1080p max.
set -x
cd "$(dirname "$0")"
Y() { yt-dlp -f 'bv*[height<=1080][ext=mp4]/bv*[height<=1080]' --no-playlist \
      --download-sections "*00:00:30-00:02:00" -o "$1" "$2" && \
      ffmpeg -y -i "$1" -an -c:v libx264 -preset fast -crf 23 "${1%.mp4}_clean.mp4" && mv "${1%.mp4}_clean.mp4" "$1"; }
Y front/front_dashcam.mp4 "ytsearch1:highway driving dashcam raw footage 1080p" &
Y rear/rear_cam.mp4       "ytsearch1:rear dash cam footage highway driving" &
Y carla/carla_front.mp4   "ytsearch1:CARLA simulator autopilot front camera driving town" &
wait
ls -lh front/ rear/ carla/
