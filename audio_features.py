"""Extracción de características de audio para detección de Wake Word.

Convierte señales de audio de 16 kHz en representaciones espectrales (Log-Mel Spectrogram),
que son la entrada adecuada para las neuronas artificiales del proyecto Hey BIS.
"""

from pathlib import Path
import wave
import numpy as np
from scipy.signal import spectrogram


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Convierte frecuencia de Hertz a la escala Mel."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convierte frecuencia de escala Mel a Hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    n_mels: int = 32,
    n_fft: int = 512,
    sample_rate: int = 16_000,
    f_min: float = 60.0,
    f_max: float = 7600.0,
) -> np.ndarray:
    """Crea un banco de filtros triangulares en escala Mel."""
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Índices de bins de la FFT
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft // 2)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left = bin_indices[m - 1]
        f_center = bin_indices[m]
        f_right = bin_indices[m + 1]

        if f_center > f_left:
            for k in range(f_left, f_center):
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        if f_right > f_center:
            for k in range(f_center, f_right):
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    return filterbank


def load_wav(path: str | Path, target_rate: int = 16_000) -> np.ndarray:
    """Carga un archivo WAV mono a 16 kHz como arreglo float32 en [-1.0, 1.0]."""
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sampwidth = audio.getsampwidth()
        framerate = audio.getframerate()
        n_frames = audio.getnframes()
        comptype = audio.getcomptype()

        if (channels, sampwidth, framerate, comptype) != (1, 2, target_rate, "NONE"):
            raise ValueError(
                f"Formato no compatible en {path}. "
                f"Se requiere mono, 16 bits, {target_rate} Hz (actual: ch={channels}, w={sampwidth}, rate={framerate})."
            )

        raw = audio.readframes(n_frames)

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples


def pad_or_crop(samples: np.ndarray, target_length: int = 16_000) -> np.ndarray:
    """Ajusta la señal a exactamente target_length muestras (1.0 s por defecto)."""
    current_length = len(samples)
    if current_length == target_length:
        return samples.copy()
    if current_length > target_length:
        # Recorta centrado
        start = (current_length - target_length) // 2
        return samples[start : start + target_length].copy()
    
    # Rellena con ceros centrado
    pad_left = (target_length - current_length) // 2
    pad_right = target_length - current_length - pad_left
    return np.pad(samples, (pad_left, pad_right), mode="constant")


def compute_log_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int = 16_000,
    n_mels: int = 32,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """Calcula el espectrograma Log-Mel para la señal dada.

    Retorna una matriz de dimensiones (n_mels, frames_temporales).
    """
    noverlap = win_length - hop_length
    _, _, sxx = spectrogram(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=win_length,
        noverlap=noverlap,
        nfft=n_fft,
        mode="magnitude",
    )

    filterbank = build_mel_filterbank(
        n_mels=n_mels,
        n_fft=n_fft,
        sample_rate=sample_rate,
    )

    # Multiplicación matricial: (n_mels, n_freqs) @ (n_freqs, n_frames) -> (n_mels, n_frames)
    mel_spec = np.matmul(filterbank, sxx)

    # Escala logarítmica (dB)
    log_mel = np.log(mel_spec + 1e-6)

    # Normalización por z-score estándar (media 0, varianza 1)
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-6
    normalized = (log_mel - mean) / std

    return normalized.astype(np.float32)


def extract_features(
    samples: np.ndarray,
    target_length: int = 16_000,
    n_mels: int = 32,
) -> np.ndarray:
    """Pipeline completo: ajusta duración (1s) y genera vector 1D aplanado de características."""
    adjusted = pad_or_crop(samples, target_length=target_length)
    mel_spec = compute_log_mel_spectrogram(adjusted, n_mels=n_mels)
    return mel_spec.flatten()
