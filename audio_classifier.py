"""Inferencia con pesos entrenados y una politica compartida de activacion."""

import argparse
import json
from pathlib import Path
import wave

import numpy as np

import audio_features
from neural_network import WakeWordNeuralNetwork


class ActivationGate:
    """Cuenta un evento por racha positiva; exige liberacion y tiempo de espera."""

    def __init__(self, cooldown: float = 1.0):
        if not np.isfinite(cooldown) or cooldown < 0:
            raise ValueError("cooldown debe ser finito y no negativo.")
        self.cooldown = cooldown
        self.active = False
        self.last_event = float("-inf")

    def update(self, is_wake: bool, time_sec: float) -> bool:
        event = bool(is_wake and not self.active and time_sec - self.last_event >= self.cooldown)
        self.active = bool(is_wake)
        if event:
            self.last_event = time_sec
        return event


class AudioNoiseClassifier:
    def __init__(self, weights_path: str | Path | None = None, wake_threshold: float | None = None):
        if weights_path is None:
            raise ValueError("Indica --weights con un modelo entrenado.")
        if not Path(weights_path).is_file():
            raise FileNotFoundError(f"No existe el modelo: {weights_path}")
        self.model = WakeWordNeuralNetwork.from_weights(weights_path)
        self.wake_threshold = self.model.wake_threshold if wake_threshold is None else wake_threshold
        if not np.isfinite(self.wake_threshold) or not 0 < self.wake_threshold <= 1:
            raise ValueError("El umbral debe estar en (0, 1].")

    def classify_file(self, wav_path: str | Path) -> dict:
        samples = audio_features.load_wav(wav_path)
        if len(samples) > audio_features.WINDOW_SAMPLES:
            raise ValueError("El archivo supera un segundo. Usa --scan para analizarlo completo.")
        return self.classify_samples(samples)

    def classify_samples(self, samples: np.ndarray) -> dict:
        samples = audio_features.validate_samples(samples)
        if len(samples) > audio_features.WINDOW_SAMPLES:
            raise ValueError("Clasifica ventanas de hasta un segundo; usa scan_streaming_window.")
        return self.model.classify_audio_samples(samples, wake_threshold=self.wake_threshold)

    def scan_streaming_window(self, long_samples: np.ndarray, window_size: int = 16_000,
                              hop_size: int = 4_000, cooldown: float = 1.0) -> list[dict]:
        if window_size != audio_features.WINDOW_SAMPLES:
            raise ValueError("Este modelo utiliza ventanas de 16000 muestras.")
        gate = ActivationGate(cooldown)
        detections = []
        for start, end, window in audio_features.audio_windows(long_samples, hop_size):
            result = self.classify_samples(window)
            time_sec = end / audio_features.SAMPLE_RATE
            detections.append({
                "time_start_sec": start / audio_features.SAMPLE_RATE,
                "time_end_sec": time_sec,
                **result,
                "activation": gate.update(result["is_wake_word"], time_sec),
            })
        return detections


def main():
    parser = argparse.ArgumentParser(description="Clasificar WAV con pesos entrenados de Hey BIS")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--scan", action="store_true", help="Analizar todo un archivo largo")
    parser.add_argument("--hop-size", type=int, default=4000)
    parser.add_argument("--cooldown", type=float, default=1.0)
    args = parser.parse_args()
    try:
        classifier = AudioNoiseClassifier(args.weights, args.threshold)
        result = (classifier.scan_streaming_window(audio_features.load_wav(args.wav),
                  hop_size=args.hop_size, cooldown=args.cooldown) if args.scan
                  else classifier.classify_file(args.wav))
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, EOFError, wave.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
