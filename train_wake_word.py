"""Entrenamiento reproducible; test se evalua con evaluate_wake_word.py."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import audio_features
from main import inspect_dataset, load_config
from neural_network import WakeWordNeuralNetwork, AdamOptimizer, SoftmaxCrossEntropy, LABEL_TO_IDX, LABELS, threshold_predictions


def validate_training_dataset(data_dir: Path) -> dict:
    report = inspect_dataset(data_dir, load_config())
    if report['errors']:
        raise ValueError(f"Dataset invalido ({len(report['errors'])} errores): " + '; '.join(report['errors'][:5]))
    if report['training_missing']:
        raise ValueError('Faltan grabaciones por particion/clase/idioma: ' + ', '.join(report['training_missing']))
    return report


def read_rows(data_dir: Path, split: str) -> list[dict]:
    with (data_dir / 'manifest.csv').open(encoding='utf-8-sig', newline='') as source:
        return [row for row in csv.DictReader(source) if row['split'] == split]


def load_dataset_split(data_dir: Path, target_split: str,
                       max_samples_per_class: int | None = None) -> tuple:
    root = data_dir.resolve(strict=True)
    if max_samples_per_class is not None and max_samples_per_class <= 0:
        raise ValueError('max_samples_per_class debe ser positivo.')
    features, targets, paths = [], [], []
    counts = Counter()
    for row in read_rows(root, target_split):
        label = row['label']
        if label not in LABEL_TO_IDX:
            raise ValueError(f'Etiqueta desconocida: {label}')
        path = (root / row['path']).resolve(strict=True)
        if Path(row['path']).is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Ruta fuera del dataset: {row['path']}")
        samples = audio_features.load_wav(path)
        if label == 'wake_word':
            if len(samples) > 16000:
                raise ValueError(f"Recorta la frase completa a un clip de hasta 1 segundo: {row['path']}")
            windows = [(0, len(samples), audio_features.pad_or_crop(samples))]
        else:
            windows = audio_features.audio_windows(samples, hop_size=16000)
        for start, end, window in windows:
            if max_samples_per_class is not None and counts[label] >= max_samples_per_class:
                break
            features.append(audio_features.extract_features(window))
            targets.append(LABEL_TO_IDX[label])
            paths.append(f"{row['path']}#{start}:{end}")
            counts[label] += 1
    if not features:
        return np.empty((0, 3136), np.float32), np.empty(0, dtype=int), []
    return np.asarray(features, np.float32), np.asarray(targets, dtype=int), paths


def metrics_from_probabilities(probs: np.ndarray, y: np.ndarray, threshold: float = 0.6) -> dict:
    if not len(y):
        raise ValueError('No hay ejemplos para evaluar.')
    preds = threshold_predictions(probs, threshold)
    confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
    for truth, pred in zip(y, preds):
        confusion[LABELS[int(truth)]][LABELS[int(pred)]] += 1
    tp = int(np.sum((y == 2) & (preds == 2)))
    fn = int(np.sum((y == 2) & (preds != 2)))
    fp = int(np.sum((y != 2) & (preds == 2)))
    tn = int(np.sum((y != 2) & (preds != 2)))
    return {'accuracy': float(np.mean(y == preds)), 'total_samples': len(y),
            'correct_predictions': int(np.sum(y == preds)), 'confusion_matrix': confusion,
            'wake_threshold': float(threshold), 'true_positives': tp, 'false_negatives': fn,
            'false_positives': fp, 'true_negatives': tn,
            'wake_recall': tp / (tp + fn) if tp + fn else None,
            'wake_precision': tp / (tp + fp) if tp + fp else None}


def evaluate_model(model: WakeWordNeuralNetwork, x: np.ndarray, y: np.ndarray,
                   wake_threshold: float | None = None) -> dict:
    threshold = model.wake_threshold if wake_threshold is None else wake_threshold
    if not len(x):
        raise ValueError('No hay ejemplos para evaluar.')
    # Limita activaciones temporales incluso con un conjunto grande.
    probs = np.concatenate([model.predict_proba(x[i:i + 128]) for i in range(0, len(x), 128)])
    return metrics_from_probabilities(probs, y, threshold)


def train_network(data_dir: Path, epochs: int = 15, batch_size: int = 32,
                  lr: float = 0.001, max_per_class: int | None = None,
                  save_path: Path = Path('wake_word_model.npz'), seed: int = 42,
                  max_false_positive_rate: float = 0.01) -> dict:
    if (epochs <= 0 or batch_size <= 0 or not np.isfinite(lr) or lr <= 0
            or not np.isfinite(max_false_positive_rate) or not 0 <= max_false_positive_rate <= 1):
        raise ValueError('Epocas, batch y lr deben ser positivos; tasa objetivo entre 0 y 1.')
    save_path = Path(save_path)
    if save_path.suffix.lower() != '.npz':
        raise ValueError('El archivo de pesos debe terminar en .npz.')
    dataset_report = validate_training_dataset(data_dir)
    print('Cargando train y validation; test queda reservado para evaluacion final.')
    x_train, y_train, paths_train = load_dataset_split(data_dir, 'train', max_per_class)
    x_val, y_val, paths_val = load_dataset_split(data_dir, 'validation')
    for name, y in [('train', y_train), ('validation', y_val)]:
        if set(y.tolist()) != {0, 1, 2}:
            raise ValueError(f'{name} requiere ejemplos de las tres clases.')
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = WakeWordNeuralNetwork()
    optimizer = AdamOptimizer(model.dense_layers, learning_rate=lr)
    history = []
    best_loss = float('inf')
    best_weights = None
    best_epoch = 0
    groups = [np.flatnonzero(y_train == k) for k in range(3)]
    balanced_size = max(map(len, groups))
    for epoch in range(1, epochs + 1):
        # Sobremuestreo solo en train: evita que el fondo domine la funcion de perdida.
        order = np.concatenate([rng.choice(g, balanced_size, replace=len(g) < balanced_size) for g in groups])
        rng.shuffle(order)
        loss_sum = 0.0
        start = time.perf_counter()
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            loss_sum += model.train_step(x_train[batch], y_train[batch], optimizer) * len(batch)
        val_probs = np.concatenate([model.predict_proba(x_val[i:i + 128]) for i in range(0, len(x_val), 128)])
        val_loss = SoftmaxCrossEntropy.loss(val_probs, y_val)
        if not np.isfinite(val_loss) or not np.isfinite(loss_sum):
            raise ValueError('Entrenamiento inestable: perdida no finita. Reduce --lr.')
        item = {'epoch': epoch, 'train_loss': loss_sum / len(order), 'validation_loss': val_loss,
                'validation_accuracy': float(np.mean(np.argmax(val_probs, axis=1) == y_val)),
                'seconds': time.perf_counter() - start}
        history.append(item)
        print(json.dumps(item), flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights = [(l.weights.copy(), l.bias.copy()) for l in model.dense_layers]
    for layer, (w, b) in zip(model.dense_layers, best_weights):
        layer.weights, layer.bias = w, b
    probs = np.concatenate([model.predict_proba(x_val[i:i + 128]) for i in range(0, len(x_val), 128)])
    # Escoge umbral con validation. Esta tasa por ventana NO es falsas alarmas por hora.
    candidates = []
    for threshold in np.linspace(0.05, 1.0, 96):
        metrics = metrics_from_probabilities(probs, y_val, float(threshold))
        fpr = metrics['false_positives'] / int(np.sum(y_val != 2))
        if fpr <= max_false_positive_rate:
            candidates.append((metrics['wake_recall'], -fpr, float(threshold)))
    if not candidates:
        raise ValueError('No se encontro un umbral con la tasa objetivo de validacion.')
    model.wake_threshold = max(candidates)[2]
    model.metadata = {'seed': seed, 'best_epoch': best_epoch,
                      'dataset_sha256': dataset_report['dataset_sha256'],
                      'manifest_sha256': hashlib.sha256((data_dir / 'manifest.csv').read_bytes()).hexdigest()}
    model.save_weights(save_path)
    report = {'schema_version': 1, 'seed': seed, 'epochs': epochs, 'batch_size': batch_size,
              'learning_rate': lr, 'max_per_class_train': max_per_class,
              'validation_fpr_target': max_false_positive_rate, 'best_epoch': best_epoch,
              'model': model.summary(), 'manifest_sha256': model.metadata['manifest_sha256'],
              'dataset_sha256': model.metadata['dataset_sha256'],
              'history': history, 'train_windows': len(y_train), 'validation_windows': len(y_val),
              'train_paths': paths_train, 'validation_paths': paths_val,
              'validation': evaluate_model(model, x_val, y_val), 'test_evaluated': False}
    report_path = save_path.with_suffix('.training.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Pesos: {save_path}; historial: {report_path}; umbral: {model.wake_threshold:.2f}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save-path', type=Path, default=Path('output/wake_word_model.npz'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-per-class', type=int, help='Limite opcional solo de train')
    parser.add_argument('--max-false-positive-rate', type=float, default=0.01)
    args = parser.parse_args()
    try:
        train_network(args.data_dir, args.epochs, args.batch_size, args.lr, args.max_per_class,
                      args.save_path, args.seed, args.max_false_positive_rate)
    except (OSError, ValueError, KeyError, EOFError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
