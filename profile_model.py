"""Herramienta para auditar el tamaño, memoria y huella de cómputo del modelo Hey BIS.

Analiza:
  1. Cantidad exacta de parámetros entrenables (pesos y sesgos).
  2. Tamaño en memoria RAM (Inferencia vs Entrenamiento con Adam).
  3. Tamaño real de los archivos de pesos en disco (.npz comprimido y estándar).
  4. Comparativa de consumo respecto a la Raspberry Pi (4 GB RAM).
  5. Latencia promedio y rendimiento de inferencia (CPU).
"""

from __future__ import annotations
from pathlib import Path
import sys
import tempfile
import time
import numpy as np

# Asegurar que los módulos de la raíz sean importables
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from neural_network import WakeWordNeuralNetwork, LABELS


def format_bytes(num_bytes: int | float) -> str:
    """Formatea bytes a KB o MB con formato legible."""
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.3f} MB"


def inspect_model_size(
    input_dim: int = 3136,
    hidden_dim1: int = 64,
    hidden_dim2: int = 32,
    num_classes: int = 3,
    pi_ram_bytes: int = 4_000_000_000,
) -> dict:
    """Calcula todas las métricas de tamaño y huella del modelo."""
    model = WakeWordNeuralNetwork(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_classes=num_classes,
    )

    layers_info = []
    total_weights = 0
    total_biases = 0

    for layer in model.dense_layers:
        w_size = layer.weights.size
        b_size = layer.bias.size
        total_weights += w_size
        total_biases += b_size
        layers_info.append({
            "name": layer.name,
            "weights_shape": layer.weights.shape,
            "biases_shape": layer.bias.shape,
            "params": w_size + b_size,
            "float32_bytes": (w_size + b_size) * 4,
        })

    total_params = total_weights + total_biases
    weights_f32_bytes = total_params * 4
    weights_f16_bytes = total_params * 2
    weights_int8_bytes = total_params * 1

    # Memoria en entrenamiento: modelo base + momentos m y v de Adam (x3) + gradientes (x2)
    # Total en entrenamiento ≈ 4 x parámetros (pesos + gradientes + m + v = 4 estados de 4 bytes)
    adam_training_bytes = total_params * 4 * 4

    # Memoria de activaciones intermedias durante 1 segundo de audio (inferencia)
    activation_bytes = (input_dim + hidden_dim1 + hidden_dim2 + num_classes) * 4

    # Medición real en disco guardando en archivo temporal
    with tempfile.TemporaryDirectory() as temp_dir:
        compressed_path = Path(temp_dir) / "model_compressed.npz"
        raw_path = Path(temp_dir) / "model_raw.npz"

        # Comprimido (np.savez_compressed)
        model.save_weights(compressed_path)
        disk_compressed_bytes = compressed_path.stat().st_size

        # Sin comprimir (np.savez estándar)
        np.savez(
            str(raw_path),
            w1=model.dense1.weights,
            b1=model.dense1.bias,
            w2=model.dense2.weights,
            b2=model.dense2.bias,
            w3=model.dense3.weights,
            b3=model.dense3.bias,
        )
        disk_raw_bytes = raw_path.stat().st_size

    # Benchmark de velocidad de inferencia (100 pasadas con 1 segundo de audio)
    fake_input = np.random.randn(1, input_dim).astype(np.float32)
    warmup = 10
    for _ in range(warmup):
        _ = model.predict_proba(fake_input)

    runs = 100
    start = time.perf_counter()
    for _ in range(runs):
        _ = model.predict_proba(fake_input)
    total_time = time.perf_counter() - start
    avg_latency_ms = (total_time / runs) * 1000.0

    return {
        "architecture": {
            "input_dim": input_dim,
            "hidden_dim1": hidden_dim1,
            "hidden_dim2": hidden_dim2,
            "num_classes": num_classes,
            "classes": LABELS,
        },
        "layers": layers_info,
        "total_parameters": total_params,
        "total_weights": total_weights,
        "total_biases": total_biases,
        "ram_inference_f32": weights_f32_bytes,
        "ram_inference_f16": weights_f16_bytes,
        "ram_inference_int8": weights_int8_bytes,
        "ram_activations_per_sample": activation_bytes,
        "ram_training_adam": adam_training_bytes,
        "disk_size_compressed": disk_compressed_bytes,
        "disk_size_raw": disk_raw_bytes,
        "pi_ram_percent": (weights_f32_bytes / pi_ram_bytes) * 100.0,
        "avg_latency_ms": avg_latency_ms,
    }



def inspect_storage_progress() -> dict:
    """Calcula el peso en almacenamiento de data, audios y procesamiento vs la meta de 10 GB."""
    import json
    config_file = ROOT_DIR / "config.json"
    target_bytes = 10_000_000_000
    if config_file.is_file():
        try:
            with config_file.open(encoding="utf-8") as f:
                target_bytes = json.load(f).get("dataset_target_bytes", 10_000_000_000)
        except Exception:
            pass

    def get_dir_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    data_dir = ROOT_DIR / "data"
    audio_dir = data_dir / "audio"
    audio_files_count = len(list(audio_dir.rglob("*.wav"))) if audio_dir.exists() else 0

    audio_bytes = get_dir_bytes(audio_dir)
    data_total_bytes = get_dir_bytes(data_dir)

    # Procesamiento y código: scripts .py, configs, modelos .npz
    processing_files = [f for f in ROOT_DIR.glob("*") if f.is_file() and f.suffix in {".py", ".json", ".npz", ".txt", ".md"}]
    processing_bytes = sum(f.stat().st_size for f in processing_files)

    # Total del proyecto completo en disco
    project_total_bytes = sum(f.stat().st_size for f in ROOT_DIR.rglob("*") if f.is_file())

    # Estimación de horas de audio (a 16 kHz mono 16 bits = 32,000 bytes/segundo)
    audio_hours_approx = (audio_bytes / 32_000) / 3600.0 if audio_bytes > 0 else 0.0
    target_hours_approx = (target_bytes / 32_000) / 3600.0

    percent = (data_total_bytes / target_bytes) * 100.0
    remaining_bytes = max(0, target_bytes - data_total_bytes)

    return {
        "audio_files_count": audio_files_count,
        "audio_bytes": audio_bytes,
        "data_total_bytes": data_total_bytes,
        "processing_bytes": processing_bytes,
        "project_total_bytes": project_total_bytes,
        "target_bytes": target_bytes,
        "progress_percent": percent,
        "remaining_bytes": remaining_bytes,
        "audio_hours_approx": audio_hours_approx,
        "target_hours_approx": target_hours_approx,
    }


def print_report(metrics: dict, storage: dict):
    """Muestra un reporte visual de la huella del modelo en consola."""
    print("=" * 68)
    print("           REPORTE DE TAMAÑO Y PESO DEL MODELO HEY BIS           ")
    print("=" * 68)

    arch = metrics["architecture"]
    print(f"\n[1] ARQUITECTURA DE LA RED:")
    print(f"    - Entrada:            {arch['input_dim']} bins (Log-Mel: 32 bandas x 98 frames)")
    print(f"    - Capa Oculta 1:      {arch['hidden_dim1']} neuronas (ReLU)")
    print(f"    - Capa Oculta 2:      {arch['hidden_dim2']} neuronas (ReLU)")
    print(f"    - Capa de Salida:     {arch['num_classes']} neuronas (Softmax: {', '.join(arch['classes'])})")

    print(f"\n[2] DESGLOSE DE PARÁMETROS POR CAPA:")
    for layer in metrics["layers"]:
        w_shape = f"{layer['weights_shape'][0]}x{layer['weights_shape'][1]}"
        b_shape = f"{layer['biases_shape'][0]}x{layer['biases_shape'][1]}"
        print(f"    * {layer['name']:<14}: W({w_shape}) + b({b_shape}) = {layer['params']:,} params ({format_bytes(layer['float32_bytes'])})")
    print(f"    ------------------------------------------------------------")
    print(f"    TOTAL PARÁMETROS:     {metrics['total_parameters']:,} parámetros")

    print(f"\n[3] PESO EN MEMORIA RAM (Inferencia / Ejecución):")
    print(f"    - Precisión Float32:  {format_bytes(metrics['ram_inference_f32'])} ({metrics['ram_inference_f32']:,} bytes)")
    print(f"    - Precisión Float16:  {format_bytes(metrics['ram_inference_f16'])} (potencial)")
    print(f"    - Cuantizado Int8:    {format_bytes(metrics['ram_inference_int8'])} (potencial ultra-ligero)")
    print(f"    - Búfer activaciones: {format_bytes(metrics['ram_activations_per_sample'])} por inferencia")

    print(f"\n[4] PESO EN ENTRENAMIENTO (RAM con Optimizador Adam):")
    print(f"    - Pesos + Gradientes + Momentos (m, v): {format_bytes(metrics['ram_training_adam'])}")

    print(f"\n[5] PESO EN DISCO DEL MODELO:")
    print(f"    - Pesos comprimidos (.npz):   {format_bytes(metrics['disk_size_compressed'])} ({metrics['disk_size_compressed']:,} bytes)")
    print(f"    - Pesos sin comprimir (.npz): {format_bytes(metrics['disk_size_raw'])} ({metrics['disk_size_raw']:,} bytes)")

    print(f"\n[6] IMPACTO EN HARDWARE OBJETIVO (Raspberry Pi 4 GB RAM):")
    print(f"    - Ocupa el {metrics['pi_ram_percent']:.4f}% de la memoria RAM total.")
    print(f"    - Estado: Extremadamente ligero (apto para microcontroladores y SBCs).")

    print(f"\n[7] VELOCIDAD DE INFERENCIA EN CPU:")
    print(f"    - Latencia promedio:  {metrics['avg_latency_ms']:.3f} ms por segundo de audio")
    fps = 1000.0 / metrics["avg_latency_ms"] if metrics["avg_latency_ms"] > 0 else 0
    print(f"    - Rendimiento:        ~{fps:.0f} inferencias por segundo (tiempo real holgado)")

    print("\n" + "=" * 68)
    print("      [8] AUDITORÍA DE STORAGE: DATA + PROCESAMIENTO VS META 10 GB  ")
    print("=" * 68)
    audio_mb = storage['audio_bytes'] / (1024 * 1024)
    data_mb = storage['data_total_bytes'] / (1024 * 1024)
    proc_mb = storage['processing_bytes'] / (1024 * 1024)
    target_gb = storage['target_bytes'] / 1_000_000_000
    target_mb = storage['target_bytes'] / (1024 * 1024)
    remain_mb = storage['remaining_bytes'] / (1024 * 1024)
    remain_gb = storage['remaining_bytes'] / (1024 * 1024 * 1024)

    print(f"  * DATA ACTUAL:")
    print(f"    - Audios WAV ({storage['audio_files_count']:,} archivos):  {audio_mb:.2f} MB ({format_bytes(storage['audio_bytes'])})")
    print(f"    - Metadatos (manifest, licencias): {(data_mb - audio_mb):.2f} MB")
    print(f"    - Total Carpeta data/:             {data_mb:.2f} MB")

    print(f"\n  * PROCESAMIENTO E IA (Código y modelos):")
    print(f"    - Scripts (.py, neural_network, etc.): {proc_mb:.3f} MB ({storage['processing_bytes']:,} bytes)")
    print(f"    - Huella del modelo entrenable:        {format_bytes(metrics['disk_size_compressed'])}")

    print(f"\n  * META DEL PROYECTO (nanoAI.pdf / config.json):")
    print(f"    - Meta objetivo de storage:        {target_gb:.1f} GB ({target_mb:.2f} MB = {storage['target_bytes']:,} bytes)")
    print(f"    - Storage actual en data:          {data_mb:.2f} MB ({storage['progress_percent']:.2f}% de la meta)")
    print(f"    - Faltante para los 10 GB:         {remain_mb:.2f} MB (~{remain_gb:.2f} GB restantes)")
    print(f"    - Horas de audio estimadas:        ~{storage['audio_hours_approx']:.1f} h actuales vs ~{storage['target_hours_approx']:.1f} h meta")
    print("=" * 68)


if __name__ == "__main__":
    metrics = inspect_model_size()
    storage = inspect_storage_progress()
    print_report(metrics, storage)

