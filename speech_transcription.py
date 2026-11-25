"""Local multilingual transcription shared by the kiosk and Python listener."""

from pathlib import Path
import os
import platform
import re
import threading

import numpy as np

MODEL = None
MODEL_LOCK = threading.Lock()


def _select_whisper_config():
    machine = platform.machine().lower()
    is_arm_pi = machine.startswith(("arm", "aarch64"))
    model_name = os.environ.get("WHISPER_MODEL", "tiny" if is_arm_pi else "base")
    cpu_threads = int(os.environ.get("WHISPER_THREADS", "2" if is_arm_pi else "4"))
    beam_size = int(os.environ.get("WHISPER_BEAM_SIZE", "1" if is_arm_pi else "5"))
    return model_name, cpu_threads, beam_size


def transcribe_samples(samples):
    global MODEL
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Invalid mono audio samples.")
    if not samples.size or np.sqrt(np.mean(samples * samples)) < 0.001:
        return {"text": "", "language": "es-MX"}
    model_name, cpu_threads, beam_size = _select_whisper_config()
    with MODEL_LOCK:
        if MODEL is None:
            from faster_whisper import WhisperModel

            MODEL = WhisperModel(
                model_name, device="cpu", compute_type="int8", cpu_threads=cpu_threads,
                download_root=str(Path(__file__).resolve().parent / "output" / "whisper"),
            )
        segments, info = MODEL.transcribe(
            samples, task="transcribe", beam_size=beam_size, vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt="Hey Bis. Hola Bis. Oye Bis.",
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    if re.fullmatch(r"(?:hola|oye)\s+bis[.!?]*", text, re.IGNORECASE):
        return {"text": text, "language": "es-MX"}
    if info.language not in {"en", "es"}:
        return {"text": "", "language": "es-MX"}
    return {"text": text, "language": "en-US" if info.language == "en" else "es-MX"}
