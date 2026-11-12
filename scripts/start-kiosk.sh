#!/bin/bash
# ==============================================================================
# BIS VOICE ASSISTANT — RASPBERRY PI KIOSK LAUNCH SCRIPT
# ==============================================================================
# Disables screen saver / blanking and launches Chromium in fullscreen kiosk mode.
# ==============================================================================

# 1. Disable screen sleep and power management
xset s noblank
xset s off
xset -dpms

# 2. Hide mouse cursor after 2 seconds of inactivity (apt install unclutter)
if command -v unclutter &> /dev/null; then
    unclutter -idle 2 -root &
fi

# 3. Determine directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
KIOSK_URL="file://${DIR}/index.html?mode=kiosk"

# 4. Optional: If a local HTTP server is preferred:
# cd "${DIR}" && python3 -m http.server 8000 &
# KIOSK_URL="http://localhost:8000/?mode=kiosk"

echo "Starting Bis Voice Assistant in Kiosk Mode..."
echo "Target URL: ${KIOSK_URL}"

# 5. Launch Chromium in hardware-accelerated kiosk mode
chromium-browser \
    --noerrdialogs \
    --disable-infobars \
    --kiosk \
    --check-for-update-interval=31536000 \
    --disable-translate \
    --enable-features=OverlayScrollbar \
    --enable-gpu-rasterization \
    --enable-zero-copy \
    --ignore-gpu-blocklist \
    --autoplay-policy=no-user-gesture-required \
    "${KIOSK_URL}"

