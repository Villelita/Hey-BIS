"""Local multilingual transcription shared by the kiosk and Python listener."""

from pathlib import Path
import re
import threading

import numpy as np

MODEL = None
MODEL_LOCK = threading.Lock()


def transcribe_samples(samples):
    global MODEL
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Invalid mono audio samples.")
    if not samples.size or np.sqrt(np.mean(samples * samples)) < 0.001:
        return {"text": "", "language": "es-MX"}
    with MODEL_LOCK:
        if MODEL is None:
            from faster_whisper import WhisperModel

            MODEL = WhisperModel(
                "base", device="cpu", compute_type="int8", cpu_threads=4,
                download_root=str(Path(__file__).resolve().parent / "output" / "whisper"),
            )
        segments, info = MODEL.transcribe(
            samples, task="transcribe", beam_size=5, vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt="Hey Bis. Hola Bis. Oye Bis.",
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    if re.fullmatch(r"(?:hola|oye)\s+bis[.!?]*", text, re.IGNORECASE):
        return {"text": text, "language": "es-MX"}
    if info.language not in {"en", "es"}:
        return {"text": "", "language": "es-MX"}
    return {"text": text, "language": "en-US" if info.language == "en" else "es-MX"}
