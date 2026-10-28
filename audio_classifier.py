"""Clasificador de audio para inferencia y análisis de ruidos / Wake Word.

Carga un modelo entrenado (o inicializado) y clasifica archivos de audio WAV o buffers
de audio en tiempo real, evaluando si es ruido de fondo, voz común o 'Hey BIS'.
"""

from pathlib import Path
from typing import Dict, Any, Union
import numpy as np

import audio_features
from neural_network import WakeWordNeuralNetwork, LABELS


class AudioNoiseClassifier:
    """Clasificador de alto nivel para inferencia de Wake Word y análisis de ruido."""

    def __init__(
        self,
        weights_path: Union[str, Path, None] = None,
        wake_threshold: float = 0.60,
    ):
        self.model = WakeWordNeuralNetwork()
        self.wake_threshold = wake_threshold
        if weights_path and Path(weights_path).is_file():
            self.model.load_weights(weights_path)

    def classify_file(self, wav_path: Union[str, Path]) -> Dict[str, Any]:
        """Carga un archivo WAV (16 kHz) y clasifica su contenido."""
        samples = audio_features.load_wav(wav_path)
        return self.classify_samples(samples)

    def classify_samples(self, samples: np.ndarray) -> Dict[str, Any]:
        """Clasifica un arreglo numpy de muestras de audio mono a 16 kHz."""
        result = self.model.classify_audio_samples(
            samples,
            wake_threshold=self.wake_threshold,
        )
        return result

    def scan_streaming_window(
        self,
        long_samples: np.ndarray,
        window_size: int = 16_000,
        hop_size: int = 8_000,
    ) -> list[Dict[str, Any]]:
        """Analiza un audio largo mediante ventana deslizante para detectar activaciones en el tiempo.
        
        Útil para medir: 'how many false wake-ups per hour of ordinary talk' (métrica del PDF).
        """
        detections = []
        total_len = len(long_samples)
        for start in range(0, max(1, total_len - window_size + 1), hop_size):
            end = start + window_size
            window = long_samples[start:end]
            if len(window) < window_size:
                window = audio_features.pad_or_crop(window, target_length=window_size)
            res = self.classify_samples(window)
            detections.append({
                "time_start_sec": round(start / 16_000, 2),
                "time_end_sec": round(end / 16_000, 2),
                **res,
            })
        return detections


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Clasificar audio con la red neuronal de Hey BIS")
    parser.add_argument("--wav", type=Path, help="Ruta al archivo WAV a clasificar")
    parser.add_argument("--weights", type=Path, default=None, help="Archivo .npz con los pesos entrenados")
    parser.add_argument("--threshold", type=float, default=0.60, help="Umbral para activar wake_word")
    args = parser.parse_args()

    classifier = AudioNoiseClassifier(weights_path=args.weights, wake_threshold=args.threshold)

    if args.wav:
        analysis = classifier.classify_file(args.wav)
        print(f"Resultado de: {args.wav}")
        print(json.dumps(analysis, indent=2))
    else:
        print("Modelo listo. Arquitectura:")
        print(json.dumps(classifier.model.summary(), indent=2))
