"""Grabar datos con permiso o escuchar Hey BIS en PC y Raspberry Pi."""

import argparse
from collections import deque
import csv
import json
from pathlib import Path
import platform
import queue
import re
import time
from uuid import uuid4
import wave

import numpy as np

from audio_classifier import ActivationGate, AudioNoiseClassifier
from import_esc50 import write_csv
from main import FIELDS, initialize


def sounddevice_module():
    try:
        import sounddevice
        return sounddevice
    except (ImportError, OSError) as error:
        raise RuntimeError("Instala requirements-audio.txt; en Raspberry Pi instala libportaudio2. "
                           + str(error)) from error


def validate_recording(data_dir, label, language, speaker, split, seconds):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", speaker):
        raise ValueError("speaker debe ser un identificador simple, por ejemplo p01.")
    if not np.isfinite(seconds) or not 0 < seconds <= 300:
        raise ValueError("La duracion debe estar en (0, 300] segundos.")
    if label == "wake_word" and seconds > 1:
        raise ValueError("Graba la frase completa en un clip de hasta 1 segundo.")
    if language not in ({"none"} if label == "background" else {"es", "en"}):
        raise ValueError("Usa es/en para voz y none para fondo.")
    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        return []
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Manifiesto incompatible.")
        rows = list(reader)
    if any(r["speaker_id"] == speaker and r["split"] != split for r in rows):
        raise ValueError("Ese hablante ya pertenece a otra particion; no mezcles train y test.")
    return rows


def record(args):
    if not args.consent:
        raise ValueError("Confirma el permiso claro de quienes se grabaran usando --consent.")
    # Validar antes de acceder al microfono o crear archivos.
    rows = validate_recording(args.data_dir, args.label, args.language,
                              args.speaker, args.split, args.seconds)
    sd = sounddevice_module()
    print(f"Grabando {args.seconds}s. Pronuncia la frase al comenzar." , flush=True)
    samples = sd.rec(round(args.seconds * 16000), samplerate=16000, channels=1,
                     dtype="int16", device=args.device, blocking=True)
    status = sd.get_status()
    if status:
        raise RuntimeError(f"La captura perdio audio; repite la grabacion: {status}")
    initialize(args.data_dir)
    relative = Path("audio") / "recorded" / f"{args.speaker}_{args.language}_{args.label}_{uuid4().hex}.wav"
    path = args.data_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        with wave.open(target, "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(samples.astype("<i2", copy=False).tobytes())
    rows.append(dict(zip(FIELDS, (relative.as_posix(), args.label, args.language,
                                  args.speaker, args.split, "yes"))))
    write_csv(args.data_dir / "manifest.csv", FIELDS, rows)
    print(f"Grabacion registrada: {path}")


def listen(args):
    classifier = AudioNoiseClassifier(args.weights, args.threshold)
    gate = ActivationGate(args.cooldown)
    if (not 1 <= args.hop_size <= 16000 or args.seconds is not None
            and (not np.isfinite(args.seconds) or args.seconds <= 0)):
        raise ValueError("hop-size debe estar en [1,16000]; seconds debe ser positivo.")
    sd = sounddevice_module()
    blocks = queue.Queue(maxsize=32)
    dropped = []

    def callback(indata, frames, timing, status):
        if status:
            dropped.append(str(status))
        try:
            blocks.put_nowait(indata[:, 0].copy())
        except queue.Full:
            dropped.append("El procesamiento no sigue el ritmo del microfono.")

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("a", encoding="utf-8") if args.log else None
    buffer = np.empty(0, dtype=np.float32)
    processed = 0
    latencies = deque(maxlen=10000)
    window_count = 0
    processing_ms_sum = 0.0
    try:
        device = sd.query_devices(args.device, "input")
        if log:
            log.write(json.dumps({"type": "session", "platform": platform.platform(),
                "device": str(device), "threshold": classifier.wake_threshold,
                "sample_rate": 16000, "hop_samples": args.hop_size,
                "cooldown_seconds": args.cooldown, "started_unix": time.time()}) + "\n")
            log.flush()
        with sd.InputStream(samplerate=16000, channels=1, dtype="float32", device=args.device,
                            blocksize=args.hop_size, callback=callback):
            started = time.perf_counter()
            print("Escuchando Hey BIS. Ctrl+C para terminar.", flush=True)
            while args.seconds is None or time.perf_counter() - started < args.seconds:
                if dropped:
                    raise RuntimeError("Captura incompleta: " + dropped[0])
                try:
                    block = blocks.get(timeout=0.5)
                except queue.Empty:
                    continue
                buffer = np.concatenate((buffer, block))
                while len(buffer) >= 16000:
                    tick = time.perf_counter()
                    result = classifier.classify_samples(buffer[:16000])
                    latency = (time.perf_counter() - tick) * 1000
                    latencies.append(latency)
                    window_count += 1
                    processing_ms_sum += latency
                    timestamp = (processed + 16000) / 16000
                    activation = gate.update(result["is_wake_word"], timestamp)
                    item = {"time_end_sec": timestamp, **result, "activation": activation,
                            "processing_ms": latency}
                    if activation:
                        print(f"HEY BIS detectado en {timestamp:.2f}s ({result['wake_word_probability']:.3f})", flush=True)
                    if log:
                        log.write(json.dumps(item) + "\n")
                        log.flush()
                    buffer = buffer[args.hop_size:]
                    processed += args.hop_size
    except KeyboardInterrupt:
        print("Escucha terminada.")
    finally:
        if log:
            log.write(json.dumps({"type": "end", "windows": window_count,
                "mean_processing_ms": processing_ms_sum / window_count if window_count else None,
                "p95_processing_ms_last_10000_windows": float(np.percentile(list(latencies), 95)) if latencies else None,
                "capture_errors": dropped}) + "\n")
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="Listar dispositivos de audio")
    rec = commands.add_parser("record", help="Grabar y registrar un clip")
    rec.add_argument("--data-dir", type=Path, default=Path("data"))
    rec.add_argument("--label", choices=["background", "other_speech", "wake_word"], required=True)
    rec.add_argument("--language", choices=["es", "en", "none"], required=True)
    rec.add_argument("--speaker", required=True)
    rec.add_argument("--split", choices=["train", "validation", "test"], required=True)
    rec.add_argument("--seconds", type=float, default=1.0)
    rec.add_argument("--consent", action="store_true")
    live = commands.add_parser("listen", help="Demo en vivo con pesos entrenados")
    live.add_argument("--weights", type=Path, required=True)
    live.add_argument("--threshold", type=float)
    live.add_argument("--hop-size", type=int, default=4000)
    live.add_argument("--cooldown", type=float, default=1.0)
    live.add_argument("--seconds", type=float)
    live.add_argument("--log", type=Path, default=Path("output/live.jsonl"))
    for command in (rec, live):
        command.add_argument("--device", help="Nombre o indice del microfono")
    args = parser.parse_args()
    if hasattr(args, "device") and args.device and args.device.isdigit():
        args.device = int(args.device)
    try:
        if args.command == "devices":
            print(sounddevice_module().query_devices())
        elif args.command == "record":
            record(args)
        else:
            listen(args)
    except (OSError, ValueError, RuntimeError, KeyError, EOFError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
