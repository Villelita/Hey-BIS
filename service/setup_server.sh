#!/usr/bin/env bash
# ==============================================================================
# Script de Despliegue, Descarga de Dataset, Entrenamiento y Auto-Ejecución
# para Raspberry Pi / Linux (Hey-BIS)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_DIR="$(cd "$REPO_DIR/.." && pwd)"

DATASET_REPO_URL="${DATASET_REPO_URL:-https://github.com/RD224/DataSet.git}"
DATASET_DIR="${DATASET_DIR:-$PARENT_DIR/dataset}"
DATA_DIR="$DATASET_DIR/Hey-BIS"
MODEL_PATH="$REPO_DIR/wake_word_model.npz"
PORT="${PORT:-8080}"
EPOCHS="${EPOCHS:-15}"

cd "$REPO_DIR"

echo "================================================================"
echo "    AUTO-INSTALACIÓN Y ENTRENAMIENTO DE HEY-BIS (RASPBERRY PI)  "
echo "================================================================"
echo "  * Proyecto:        $REPO_DIR"
echo "  * Dataset Repo:    $DATASET_REPO_URL"
echo "  * Carpeta Dataset: $DATASET_DIR"
echo "  * Modelo Destino:  $MODEL_PATH"
echo "================================================================"

# 1. Instalar dependencias del sistema en Raspberry Pi / Debian / Ubuntu
if command -v apt-get &>/dev/null; then
    echo "[1/6] Verificando e instalando paquetes del sistema (python3, venv, git, libportaudio2, openssl)..."
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-pip python3-venv git libportaudio2 openssl curl
else
    echo "[1/6] Gestor apt-get no detectado (ejecución en macOS u otro SO). Omitiendo apt-get..."
fi

# 2. Configurar entorno virtual Python e instalar dependencias
echo "[2/6] Configurando entorno virtual Python (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-audio.txt || true

# 3. Clonar o actualizar el repositorio del Dataset automáticamente
echo "[3/6] Preparando repositorio de audios en $DATASET_DIR..."
if [ -d "$DATASET_DIR/.git" ]; then
    echo "    -> El repositorio de dataset ya existe. Actualizando cambios remotos..."
    git -C "$DATASET_DIR" pull --ff-only || echo "    [!] Aviso: Se mantienen los cambios locales del dataset."
elif [ -d "$DATASET_DIR" ] && [ -f "$DATA_DIR/manifest.csv" ]; then
    echo "    -> Carpeta de dataset local detectada en $DATASET_DIR."
else
    echo "    -> Clonando $DATASET_REPO_URL en $DATASET_DIR..."
    git clone "$DATASET_REPO_URL" "$DATASET_DIR"
fi

# 4. Sincronizar secciones de audios y convertir a 16 kHz PCM mono 16-bit
echo "[4/6] Sincronizando y preparando audios para entrenamiento..."
python "$REPO_DIR/sync_dataset.py" --dataset-repo "$DATASET_DIR"
python "$REPO_DIR/prepare_audio.py" --data-dir "$DATA_DIR"

# 5. Validar dataset y entrenar la Red Neuronal de Hey-BIS
echo "[5/6] Validando dataset y entrenando la IA..."
if python "$REPO_DIR/main.py" inspect --data-dir "$DATA_DIR" --require-training-ready >/dev/null 2>&1; then
    python "$REPO_DIR/train_wake_word.py" \
        --data-dir "$DATA_DIR" \
        --epochs "$EPOCHS" \
        --batch-size 32 \
        --lr 0.0005 \
        --save-path "$MODEL_PATH"

    mkdir -p "$REPO_DIR/output"
    cp "$MODEL_PATH" "$REPO_DIR/output/wake_word_model.npz"
    cp "${MODEL_PATH%.npz}.training.json" "$REPO_DIR/output/wake_word_model.training.json" 2>/dev/null || true

    echo "    -> Evaluando modelo sobre la partición reservada (test)..."
    python "$REPO_DIR/evaluate_wake_word.py" \
        --data-dir "$DATA_DIR" \
        --weights "$MODEL_PATH" \
        --output "$REPO_DIR/output/evaluation.json"
else
    echo "    [!] Aviso: El dataset en $DATA_DIR aún no tiene todas las particiones de wake_word."
    if [ -f "$MODEL_PATH" ]; then
        echo "    -> Se utilizará el modelo existente en $MODEL_PATH."
    else
        echo "    -> Agrega grabaciones de 'Hey BIS' en $DATA_DIR/audio/wake_word/ y vuelve a ejecutar este script."
    fi
fi

# 6. Configurar y arrancar el servicio automático systemd en Linux / Raspberry Pi
if command -v systemctl &>/dev/null && [ -d "/etc/systemd/system" ]; then
    echo "[6/6] Configurando servicio systemd (hey-bis.service) para auto-arranque..."
    CURRENT_USER="$(whoami)"
    SERVICE_PATH="/etc/systemd/system/hey-bis.service"

    WEIGHTS_ARG=""
    if [ -f "$MODEL_PATH" ]; then
        WEIGHTS_ARG="--weights $MODEL_PATH"
    fi

    sudo bash -c "cat > $SERVICE_PATH" <<EOF
[Unit]
Description=Hey BIS Wake Word & Audio AI Service
After=network.target sound.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/.venv/bin/python3 $REPO_DIR/service/server.py --host 0.0.0.0 --port $PORT $WEIGHTS_ARG
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable hey-bis.service
    sudo systemctl restart hey-bis.service

    if command -v ufw &>/dev/null; then
        sudo ufw allow "$PORT/tcp" comment 'Hey BIS HTTP Service' 2>/dev/null || true
        sudo ufw allow "8443/tcp" comment 'Hey BIS HTTPS Service (Client Mic)' 2>/dev/null || true
    fi

    echo ""
    echo "================================================================"
    echo "    ¡HEY-BIS DESPLEGADO, ENTRENADO Y ACTIVO EN RASPBERRY PI!    "
    echo "================================================================"
    sudo systemctl status hey-bis.service --no-pager
else
    echo "[6/6] Entorno sin systemd detectado. Puedes iniciar el servidor manualmente con:"
    echo "      $REPO_DIR/.venv/bin/python3 $REPO_DIR/service/server.py --weights $MODEL_PATH"
fi
