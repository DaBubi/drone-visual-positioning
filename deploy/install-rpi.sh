#!/usr/bin/env bash
# Install the VPS onboard module on a Raspberry Pi CM4/5
# Usage: ./install-rpi.sh [map_pack.tar.gz]
set -euo pipefail

VPS_DIR="/opt/vps"
VENV_DIR="$VPS_DIR/venv"
SERVICE_NAME="vps-onboard"

echo "=== Drone VPS Onboard Installer ==="

# Must run as root
if [ "$EUID" -ne 0 ]; then
    echo "Run as root: sudo ./install-rpi.sh"
    exit 1
fi

# 1. System deps
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-dev \
    libopencv-dev python3-opencv \
    libatlas-base-dev libhdf5-dev \
    2>/dev/null

# 2. Create directory structure
echo "[2/6] Setting up $VPS_DIR..."
mkdir -p "$VPS_DIR"/{maps,logs,data,models}

# 3. Python venv + install
echo "[3/6] Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip wheel 2>/dev/null

# Copy source and install
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cp -r "$SCRIPT_DIR/src" "$VPS_DIR/"
cp "$SCRIPT_DIR/pyproject.toml" "$VPS_DIR/"

echo "[4/6] Installing Python packages..."
"$VENV_DIR/bin/pip" install "$VPS_DIR[onboard]" 2>&1 | tail -1

# 4. Default config
if [ ! -f "$VPS_DIR/config.json" ]; then
    echo "[5/6] Writing default config..."
    cat > "$VPS_DIR/config.json" << 'CONF'
{
    "map_pack": "/opt/vps/maps/map_pack",
    "camera": {
        "device": 0,
        "width": 640,
        "height": 640,
        "fps": 10,
        "use_picamera2": false
    },
    "uart": {
        "port": "/dev/ttyAMA0",
        "baudrate": 9600,
        "enabled": true
    },
    "matcher": {
        "min_matches": 15,
        "confidence_threshold": 0.3,
        "max_candidates": 5,
        "use_orb_fallback": true
    },
    "target_hz": 3.0,
    "log_level": "INFO",
    "ekf_measurement_noise": 1e-8,
    "ekf_gate_threshold": 9.0,
    "telemetry_dir": "/opt/vps/logs"
}
CONF
else
    echo "[5/6] Config already exists, skipping."
fi

# 5. Install systemd service
echo "[6/6] Installing systemd service..."
cp "$SCRIPT_DIR/deploy/$SERVICE_NAME.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# 6. Load map pack if provided
if [ -n "${1:-}" ] && [ -f "$1" ]; then
    echo "Extracting map pack: $1"
    tar -xzf "$1" -C "$VPS_DIR/maps/"
fi

# Set ownership
chown -R pi:pi "$VPS_DIR"

# Add pi user to dialout for UART access
usermod -a -G dialout pi 2>/dev/null || true

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Transfer map pack:  scp map_pack.tar.gz pi@drone:/tmp/"
echo "     Then extract:       tar -xzf /tmp/map_pack.tar.gz -C /opt/vps/maps/"
echo "  2. Edit config:        nano /opt/vps/config.json"
echo "  3. Connect camera and UART"
echo "  4. Start service:      sudo systemctl start $SERVICE_NAME"
echo "  5. View logs:          journalctl -u $SERVICE_NAME -f"
echo "  6. View telemetry:     ls /opt/vps/logs/"
echo ""
echo "Betaflight UART setup:"
echo "  - Ports tab: enable GPS on the UART connected to RPi"
echo "  - Configuration tab: GPS Protocol = NMEA"
echo "  - Save and reboot FC"
