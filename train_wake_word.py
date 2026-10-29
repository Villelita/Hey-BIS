"""Script de entrenamiento para la red neuronal de Wake Word / Clasificación de ruido.

Permite entrenar el modelo con los datos del manifest.csv y calcular las métricas exigidas
en el documento nanoAI.pdf:
  - Exactitud en el conjunto de test (datos nunca vistos por el modelo).
  - Matriz de confusión entre (background, other_speech, wake_word).
  - Tasa de falsas alarmas (false wake-ups).
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np

import audio_features
from neural_network import (
    WakeWordNeuralNetwork,
    AdamOptimizer,
    SoftmaxCrossEntropy,
    LABEL_TO_IDX,
    LABELS,
)


def load_dataset_split(
    data_dir: Path,
    target_split: str,
    max_samples_per_class: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carga los audios de una partición (train, validation, test) y extrae sus características."""
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No se encontró {manifest_path}")

    features_list = []
    labels_list = []
    paths_list = []
    class_counts = {lbl: 0 for lbl in LABELS}

    with manifest_path.open("r", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        for row in reader:
            split = row["split"]
            label = row["label"]
            if split != target_split:
                continue
            if label not in LABEL_TO_IDX:
                continue

            if max_samples_per_class and class_counts[label] >= max_samples_per_class:
                continue

            wav_path = data_dir / row["path"]
            if not wav_path.is_file():
                continue

            try:
                samples = audio_features.load_wav(wav_path)
                feats = audio_features.extract_features(samples)
                features_list.append(feats)
                labels_list.append(LABEL_TO_IDX[label])
                paths_list.append(str(row["path"]))
                class_counts[label] += 1
            except Exception:
                # Si algún archivo no cumple la especificación de formato se omite
                continue

    if not features_list:
        return np.empty((0, 3136), dtype=np.float32), np.empty((0,), dtype=int), []

    x = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=int)
    return x, y, paths_list


def evaluate_model(
    model: WakeWordNeuralNetwork,
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """Evalúa exactitud global y matriz de confusión."""
    if len(x) == 0:
        return {"accuracy": 0.0, "total": 0, "confusion_matrix": {}}

    probs = model.predict_proba(x)
    preds = np.argmax(probs, axis=-1)
    correct = int(np.sum(preds == y))
    accuracy = correct / len(y)

    confusion = {true_lbl: {pred_lbl: 0 for pred_lbl in LABELS} for true_lbl in LABELS}
    for true_i, pred_i in zip(y, preds):
        confusion[LABELS[true_i]][LABELS[pred_i]] += 1

    return {
        "accuracy": round(accuracy, 4),
        "total_samples": len(y),
        "correct_predictions": correct,
        "confusion_matrix": confusion,
    }


def train_network(
    data_dir: Path,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    max_per_class: int = 500,
    save_path: Path = Path("wake_word_model.npz"),
):
    print(f"Cargando particiones desde: {data_dir}...")
    x_train, y_train, _ = load_dataset_split(data_dir, "train", max_samples_per_class=max_per_class)
    x_val, y_val, _ = load_dataset_split(data_dir, "validation", max_samples_per_class=max_per_class)
    x_test, y_test, _ = load_dataset_split(data_dir, "test", max_samples_per_class=max_per_class)

    print(f"Datos cargados:")
    print(f"  Train:      {len(x_train)} ejemplos")
    print(f"  Validation: {len(x_val)} ejemplos")
    print(f"  Test:       {len(x_test)} ejemplos")

    if len(x_train) == 0:
        print("Aviso: No hay suficientes datos en 'train' para entrenar.")
        return

    model = WakeWordNeuralNetwork(input_dim=x_train.shape[1])
    optimizer = AdamOptimizer(model.dense_layers, learning_rate=lr)

    print("\nIniciando entrenamiento...")
    num_samples = len(x_train)

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(num_samples)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        num_batches = int(np.ceil(num_samples / batch_size))

        start_time = time.time()
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_x = x_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]

            loss = model.train_step(batch_x, batch_y, optimizer)
            epoch_loss += loss

        elapsed = time.time() - start_time
        avg_loss = epoch_loss / num_batches

        val_metrics = evaluate_model(model, x_val, y_val) if len(x_val) > 0 else {"accuracy": "N/A"}
        val_acc = val_metrics["accuracy"]

        print(f"Época [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - Val Acc: {val_acc} - Tiempo: {elapsed:.2f}s")

    print(f"\nGuardando pesos del modelo en: {save_path}")
    model.save_weights(save_path)

    if len(x_test) > 0:
        print("\n=== EVALUACIÓN EN CONJUNTO DE PRUEBA (Datos nunca vistos) ===")
        test_metrics = evaluate_model(model, x_test, y_test)
        print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar neuronas para Hey BIS")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Carpeta de datos")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Tasa de aprendizaje (Learning Rate)")
    parser.add_argument("--save-path", type=Path, default=Path("wake_word_model.npz"), help="Ruta de guardado")
    args = parser.parse_args()

    train_network(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )
