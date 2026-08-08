#!/usr/bin/env bash
# Screenshot the PipeGen window whenever its content changes.
# Keeps timestamped PNGs in docs/screenshots/ for the submission README.
export DISPLAY=:1
DIR="$(dirname "$0")/screenshots"
mkdir -p "$DIR"
PREV=""
while true; do
    WID=$(xdotool search --name "^PipeGen$" 2>/dev/null | while read -r w; do
        g=$(xdotool getwindowgeometry "$w" 2>/dev/null | grep Geometry | tr -dc '0-9x')
        [ "${g%x*}" -gt 800 ] 2>/dev/null && echo "$w" && break
    done)
    if [ -n "$WID" ]; then
        TMP=$(mktemp --suffix=.png)
        if import -window "$WID" "$TMP" 2>/dev/null; then
            SUM=$(md5sum "$TMP" | cut -d' ' -f1)
            if [ "$SUM" != "$PREV" ]; then
                mv "$TMP" "$DIR/$(date +%H%M%S)_pipegen.png"
                PREV=$SUM
            else
                rm -f "$TMP"
            fi
        fi
    fi
    sleep 40
done
