"""Pruebas unitarias para el servicio web y API de Hey-BIS."""

import io
import json
import unittest
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

from service.server import (
    get_lan_ips,
    decode_audio_bytes_to_16k_mono,
    generate_synthetic_wav,
    AudioNoiseClassifier,
)


class TestService(unittest.TestCase):
    """Verifica la lógica central del servicio web."""

    def test_get_lan_ips(self):
        ips = get_lan_ips()
        self.assertIsInstance(ips, list)
        self.assertGreater(len(ips), 0)
        for ip in ips:
            self.assertIsInstance(ip, str)
            self.assertIn(".", ip)

    def test_decode_standard_16k_wav(self):
        sr = 16000
        n_samples = 16000
        t = np.linspace(0, 1.0, n_samples, endpoint=False)
        audio_int16 = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        buf = io.BytesIO()
        wavfile.write(buf, sr, audio_int16)
        raw_bytes = buf.getvalue()

        decoded = decode_audio_bytes_to_16k_mono(raw_bytes)
        self.assertEqual(len(decoded), 16000)
        self.assertEqual(decoded.dtype, np.float32)
        self.assertAlmostEqual(float(np.max(decoded)), 0.5, places=2)

    def test_decode_resample_44k_stereo_to_16k_mono(self):
        sr = 44100
        duration = 0.5
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        left = 0.4 * np.sin(2 * np.pi * 440 * t)
        right = 0.4 * np.cos(2 * np.pi * 440 * t)
        stereo_int16 = (np.column_stack([left, right]) * 32767).astype(np.int16)

        buf = io.BytesIO()
        wavfile.write(buf, sr, stereo_int16)
        raw_bytes = buf.getvalue()

        decoded = decode_audio_bytes_to_16k_mono(raw_bytes)
        # 0.5s a 16 kHz deben ser 8000 muestras
        self.assertEqual(len(decoded), 8000)
        self.assertEqual(decoded.ndim, 1)

    def test_generate_synthetic_wav(self):
        for sample_type in ("silence", "tone", "noise"):
            wav_bytes = generate_synthetic_wav(sample_type=sample_type, duration_sec=1.0)
            self.assertGreater(len(wav_bytes), 44)
            # Decodificar el WAV generado
            sr, data = wavfile.read(io.BytesIO(wav_bytes))
            self.assertEqual(sr, 16000)
            self.assertEqual(len(data), 16000)

    def test_classifier_silence_inference(self):
        classifier = AudioNoiseClassifier(weights_path=None, wake_threshold=0.60)
        silence = np.zeros(16000, dtype=np.float32)
        result = classifier.classify_samples(silence)
        self.assertIn("label", result)
        self.assertIn("probabilities", result)
        self.assertIn("wake_word", result["probabilities"])
        self.assertIn("other_speech", result["probabilities"])
        self.assertIn("background", result["probabilities"])


if __name__ == "__main__":
    unittest.main()

