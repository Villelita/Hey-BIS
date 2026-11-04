#!/usr/bin/env bash
# ==============================================================================
# Script de Despliegue y Configuración Automática en Servidor Físico (Linux/Pi)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"

echo "================================================================"
echo "    INSTALACIÓN Y DESPLIEGUE DE HEY-BIS EN SERVIDOR REMOTO     "
echo "================================================================"

# 1. Asegurar dependencias del sistema
if command -v apt-get &>/dev/null; then
    echo "[*] Actualizando paquetes del sistema e instalando dependencias..."
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-pip python3-venv git
fi

# 2. Configurar entorno virtual Python
if [ ! -d ".venv" ]; then
    echo "[*] Creando entorno virtual .venv..."
    python3 -m venv .venv
fi

echo "[*] Instalando dependencias de Hey-BIS..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configurar servicio systemd para arranque automático y persistencia
CURRENT_USER="$(whoami)"
SERVICE_PATH="/etc/systemd/system/hey-bis.service"

echo "[*] Registrando servicio systemd en $SERVICE_PATH..."
sudo bash -c "cat > $SERVICE_PATH" <<EOF
[Unit]
Description=Hey BIS Wake Word & Audio AI Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/.venv/bin/python3 $REPO_DIR/service/server.py --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 4. Habilitar e iniciar servicio
echo "[*] Recargando e iniciando servicio hey-bis..."
sudo systemctl daemon-reload
sudo systemctl enable hey-bis.service
sudo systemctl restart hey-bis.service

# 5. Si existe UFW (firewall), permitir puerto 8080
if command -v ufw &>/dev/null; then
    sudo ufw allow 8080/tcp comment 'Hey BIS Service' 2>/dev/null || true
fi

echo ""
echo "================================================================"
echo "    ¡HEY-BIS SE HA DESPLEGADO EXITOSAMENTE EN EL SERVIDOR!     "
echo "================================================================"
sudo systemctl status hey-bis.service --no-pager
