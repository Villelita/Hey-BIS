#!/usr/bin/env bash
# ==============================================================================
# Script de inicio para el Servicio Web de Hey-BIS
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 no está instalado o no se encuentra en el PATH."
    exit 1
fi

echo "[*] Iniciando Hey-BIS Service en 0.0.0.0..."
exec python3 "$SCRIPT_DIR/server.py" "$@"

