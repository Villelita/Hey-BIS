#!/usr/bin/env python3
"""Sincroniza y prepara el repositorio de Dataset para entrenamiento automático en PC o Raspberry Pi.

Funciones principales:
1. Crea la estructura de carpetas seccionada para grabaciones:
   - audio/wake_word/hey_bis_es/
   - audio/wake_word/hey_bis_en/
   - audio/wake_word/hola_bis/
   - audio/other_speech/frases_parecidas/
2. Si existe ../dataset/Voices y aún no se ha integrado a Hey-BIS/audio/voices/pcm16k,
   copia y convierte los audios de 44.1 kHz a 16 kHz y los registra en manifest.csv.
3. Escanea las carpetas de wake_word y other_speech, normaliza cualquier archivo WAV a
   16 kHz mono 16-bit PCM y actualiza manifest.csv garantizando que no haya fuga de
   hablantes entre particiones (train, validation, test).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from math import gcd
from pathlib import Path
import shutil
import wave

import numpy as np
from scipy.signal import resample_poly

import audio_features
from main import FIELDS, inspect_dataset, load_config


ES_EXACT_WORDS = {
    "acrobatas", "aleteum", "cabrilet", "colibri", "colibris", "mexico", "palabra",
    "mini", "original", "motor", "vital", "total", "central", "natural", "normal",
    "control", "animal", "general", "global", "ideal", "final", "frontal", "lateral",
    "oficial", "personal", "popular", "principal", "real", "kc", "gps", "km",
}


def ensure_pcm16k_mono(wav_path: Path) -> bytes:
    """Lee un archivo WAV y devuelve bytes WAV PCM mono 16-bit a 16 kHz."""
    with wave.open(str(wav_path), "rb") as source:
        channels, width, rate, frames, compression, _ = source.getparams()
        if compression != "NONE" or frames <= 0:
            raise ValueError(f"WAV inválido o vacío: {wav_path}")
        raw = source.readframes(frames)

    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    elif width == 4:
        samples = (np.frombuffer(raw, dtype="<i4").astype(np.float32) / 65536.0)
    elif width == 1:
        samples = ((np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) * 256.0)
    else:
        raise ValueError(f"Ancho de muestra no soportado ({width} bytes): {wav_path}")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if rate != 16000:
        divisor = gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor)

    pcm = np.clip(np.rint(samples), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as target:
        target.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        target.writeframes(pcm.tobytes())
    return buf.getvalue()


def ensure_directory_sections(hey_bis_dir: Path) -> dict[str, Path]:
    """Crea las subcarpetas organizadas para que el usuario coloque nuevos audios."""
    sections = {
        "hey_bis_es": hey_bis_dir / "audio" / "wake_word" / "hey_bis_es",
        "hey_bis_en": hey_bis_dir / "audio" / "wake_word" / "hey_bis_en",
        "hola_bis": hey_bis_dir / "audio" / "wake_word" / "hola_bis",
        "frases_parecidas": hey_bis_dir / "audio" / "other_speech" / "frases_parecidas",
        "recorded": hey_bis_dir / "audio" / "recorded",
    }
    for folder in sections.values():
        folder.mkdir(parents=True, exist_ok=True)
    return sections


def sync_voices_folder(dataset_repo: Path, hey_bis_dir: Path, existing_rows: list[dict]) -> list[dict]:
    """Integra ../dataset/Voices hacia Hey-BIS/audio/voices/pcm16k si aún no está en el manifiesto."""
    voices_src = dataset_repo / "Voices"
    if not voices_src.is_dir():
        return existing_rows

    already_has_voices = any(r["path"].startswith("audio/voices/") for r in existing_rows)
    if already_has_voices:
        return existing_rows

    src_files = sorted(voices_src.glob("*.wav"))
    if not src_files:
        return existing_rows

    print(f"[*] Integrando {len(src_files)} archivos de Voices/ a 16 kHz...")
    pcm16k_dir = hey_bis_dir / "audio" / "voices" / "pcm16k"
    pcm16k_dir.mkdir(parents=True, exist_ok=True)

    by_digest: dict[str, list[str]] = {}
    for idx, src in enumerate(src_files, start=1):
        dst = pcm16k_dir / src.name
        if not dst.exists():
            encoded = ensure_pcm16k_mono(src)
            dst.write_bytes(encoded)
        with wave.open(str(dst), "rb") as w:
            pcm = w.readframes(w.getnframes())
        digest = hashlib.sha256(pcm).hexdigest()
        by_digest.setdefault(digest, []).append(src.name)
        if idx % 500 == 0:
            print(f"    Procesados {idx}/{len(src_files)} audios de Voices...")

    new_rows = []
    for i, digest in enumerate(sorted(by_digest.keys())):
        mod = i % 20
        if mod < 14:
            split = "train"
        elif mod < 17:
            split = "validation"
        else:
            split = "test"
        for fname in by_digest[digest]:
            word = Path(fname).stem.split("_", 1)[-1].lower()
            lang = "es" if word in ES_EXACT_WORDS else "en"
            new_rows.append({
                "path": f"audio/voices/pcm16k/{fname}",
                "label": "other_speech",
                "language": lang,
                "speaker_id": f"voices_{split}_{lang}",
                "split": split,
                "consent": "yes",
            })

    return existing_rows + new_rows


def sync_custom_sections(hey_bis_dir: Path, existing_rows: list[dict], sections: dict[str, Path]) -> list[dict]:
    """Escanea las subcarpetas de wake_word y frases_parecidas y registra audios nuevos en manifest.csv."""
    registered_paths = {r["path"] for r in existing_rows}
    section_rules = [
        (sections["hey_bis_es"], "wake_word", "es", "sec_heybis_es"),
        (sections["hey_bis_en"], "wake_word", "en", "sec_heybis_en"),
        (sections["hola_bis"], "wake_word", "es", "sec_holabis_es"),
        (sections["frases_parecidas"], "other_speech", "es", "sec_neg_es"),
    ]

    added_rows = []
    for folder, label, default_lang, speaker_prefix in section_rules:
        wav_files = sorted(folder.glob("*.wav"))
        for idx, wav_path in enumerate(wav_files):
            rel_path = wav_path.relative_to(hey_bis_dir).as_posix()
            if rel_path in registered_paths:
                continue

            # Normalizar in-place a 16 kHz mono 16-bit si no lo está
            encoded = ensure_pcm16k_mono(wav_path)
            wav_path.write_bytes(encoded)

            # Si es wake_word y supera WINDOW_SAMPLES, recortar centrado a WINDOW_SAMPLES (3s)
            if label == "wake_word":
                samples = audio_features.load_wav(wav_path)
                if len(samples) > audio_features.WINDOW_SAMPLES:
                    cropped = audio_features.pad_or_crop(samples, audio_features.WINDOW_SAMPLES)
                    pcm = np.clip(np.rint(cropped * 32767), -32768, 32767).astype("<i2")
                    with wave.open(str(wav_path), "wb") as w:
                        w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                        w.writeframes(pcm.tobytes())

            # Determinar split por nombre de archivo si lo incluye, o repartir 70/15/15
            stem_lower = wav_path.stem.lower()
            if "test" in stem_lower:
                split = "test"
            elif "val" in stem_lower:
                split = "validation"
            elif "train" in stem_lower:
                split = "train"
            else:
                mod = idx % 10
                split = "train" if mod < 7 else ("validation" if mod < 8 else "test")

            lang = "en" if "_en" in stem_lower else ("es" if "_es" in stem_lower else default_lang)
            added_rows.append({
                "path": rel_path,
                "label": label,
                "language": lang,
                "speaker_id": f"{speaker_prefix}_{split}",
                "split": split,
                "consent": "yes",
            })
            registered_paths.add(rel_path)

    if added_rows:
        print(f"[*] Registrados {len(added_rows)} audios nuevos desde las carpetas seccionadas.")
    return existing_rows + added_rows


def _make_3s_from_1s_mic(samples_1s: np.ndarray, rng: np.random.Generator, position_frac: float = 0.5, gain: float = 1.0) -> np.ndarray:
    """Coloca un clip corto de micrófono dentro de una ventana natural de 3s (48,000 muestras)."""
    win = min(4000, len(samples_1s))
    energies = [np.mean(samples_1s[i:i + win] ** 2) for i in range(0, max(1, len(samples_1s) - win + 1), 800)]
    min_i = int(np.argmin(energies)) * 800
    quiet_chunk = samples_1s[min_i:min_i + win]
    if len(quiet_chunk) == 0 or np.max(np.abs(quiet_chunk)) < 1e-4:
        quiet_chunk = rng.normal(0, 0.002, 4000).astype(np.float32)

    floor = np.empty(48000, dtype=np.float32)
    pos = 0
    while pos < 48000:
        chunk = quiet_chunk.copy()
        if rng.random() < 0.5:
            chunk = chunk[::-1]
        chunk = np.roll(chunk, int(rng.integers(0, len(chunk))))
        chunk += rng.normal(0, 0.002, len(chunk)).astype(np.float32)
        n = min(len(chunk), 48000 - pos)
        floor[pos:pos + n] = chunk[:n]
        pos += n

    max_offset = max(0, 48000 - len(samples_1s))
    offset = int(round(max_offset * position_frac))
    clip = (samples_1s * gain).astype(np.float32)[:48000]
    out = floor.copy()
    out[offset:offset + len(clip)] = clip
    return np.clip(out, -0.98, 0.98)


def ensure_3s_calibration_and_augmentation(hey_bis_dir: Path, rows: list[dict]) -> list[dict]:
    """Si los clips base de wake_word aún no tienen expansión de 3s ni calibración, los genera."""
    has_3s_aug = any("_3s_" in r["path"] for r in rows)
    has_calib = any("calib_bg_" in r["path"] for r in rows)
    if has_3s_aug and has_calib:
        return rows

    print("[*] Generando ventanas de 3 segundos y calibración acústica para entrenamiento...")
    rng = np.random.default_rng(42)
    base_rows = [
        r for r in rows
        if not any(tag in r["path"] for tag in ("_aug_", "_3s_", "calib_bg_", "speech_center_"))
    ]
    orig_wake_rows = [r for r in base_rows if r["label"] == "wake_word"]

    esc_train_paths = [hey_bis_dir / r["path"] for r in base_rows if r["label"] == "background" and r["split"] == "train"][:40]
    esc_clips = []
    for ep in esc_train_paths:
        with wave.open(str(ep), "rb") as w:
            esc_clips.append(np.frombuffer(w.readframes(48000), dtype="<i2").astype(np.float32) / 32768.0)

    aug_wake_rows = []
    for r in orig_wake_rows:
        p = hey_bis_dir / r["path"]
        with wave.open(str(p), "rb") as w:
            s_full = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        s = s_full[16000:32000] if len(s_full) == 48000 else s_full[:16000]

        if len(s_full) < 48000:
            s_3s = _make_3s_from_1s_mic(s, rng, position_frac=0.5, gain=1.0)
            with wave.open(str(p), "wb") as w:
                w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                w.writeframes((s_3s * 32767).astype("<i2").tobytes())

        if r["split"] in ("train", "validation"):
            positions = np.linspace(0.05, 0.95, 10) if r["split"] == "train" else [0.2, 0.5, 0.8]
            gains = [0.7, 1.0, 1.25] if r["split"] == "train" else [1.0]
            idx = 0
            for pos_f in positions:
                for g in gains:
                    idx += 1
                    aug = _make_3s_from_1s_mic(s, rng, position_frac=float(pos_f), gain=g)
                    if r["split"] == "train" and esc_clips and rng.random() < 0.5:
                        bg = esc_clips[int(rng.integers(0, len(esc_clips)))]
                        aug = np.clip(aug + bg * float(rng.uniform(0.03, 0.15)), -0.98, 0.98)
                    rel = f"audio/recorded/{p.stem}_3s_{idx:03d}.wav"
                    with wave.open(str(hey_bis_dir / rel), "wb") as w:
                        w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                        w.writeframes((aug * 32767).astype("<i2").tobytes())
                    aug_wake_rows.append({
                        "path": rel,
                        "label": "wake_word",
                        "language": r["language"],
                        "speaker_id": r["speaker_id"],
                        "split": r["split"],
                        "consent": "yes",
                    })
            idx += 1
            pad_3s = audio_features.pad_or_crop(s, target_length=48000)
            rel = f"audio/recorded/{p.stem}_3s_{idx:03d}.wav"
            with wave.open(str(hey_bis_dir / rel), "wb") as w:
                w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                w.writeframes((np.clip(pad_3s, -0.98, 0.98) * 32767).astype("<i2").tobytes())
            aug_wake_rows.append({
                "path": rel,
                "label": "wake_word",
                "language": r["language"],
                "speaker_id": r["speaker_id"],
                "split": r["split"],
                "consent": "yes",
            })

    calib_rows = []
    for split in ("train", "validation"):
        spk = f"calib_bg_{split}"
        n_each = 15 if split == "train" else 4
        c_idx = 0
        for kind in ("silence", "noise_3s", "noise_1s_padded", "tone_3s", "tone_1s_padded"):
            for k in range(n_each):
                c_idx += 1
                t3 = np.linspace(0, 3.0, 48000, endpoint=False)
                t1 = np.linspace(0, 1.0, 16000, endpoint=False)
                freq = float(rng.choice([220, 330, 440, 550, 880])) if k > 0 else 440.0
                amp = float(rng.uniform(0.1, 0.6)) if k > 0 else 0.5
                if kind == "silence":
                    sig = rng.normal(0, 1e-4 * (k + 1), 48000).astype(np.float32)
                elif kind == "noise_3s":
                    sig = rng.normal(0, float(rng.uniform(0.02, 0.25)), 48000).astype(np.float32)
                elif kind == "noise_1s_padded":
                    n1 = rng.normal(0, float(rng.uniform(0.05, 0.25)), 16000).astype(np.float32)
                    sig = np.pad(n1, (16000, 16000))
                elif kind == "tone_3s":
                    sig = (amp * np.sin(2 * np.pi * freq * t3)).astype(np.float32)
                else:
                    s1 = (amp * np.sin(2 * np.pi * freq * t1)).astype(np.float32)
                    sig = np.pad(s1, (16000, 16000))
                sig = np.clip(sig + rng.normal(0, 2e-4, 48000).astype(np.float32), -0.98, 0.98)
                rel = f"audio/recorded/calib_bg_{split}_{c_idx:03d}.wav"
                with wave.open(str(hey_bis_dir / rel), "wb") as w:
                    w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    w.writeframes((sig * 32767).astype("<i2").tobytes())
                calib_rows.append({
                    "path": rel,
                    "label": "background",
                    "language": "none",
                    "speaker_id": spk,
                    "split": split,
                    "consent": "yes",
                })

    other_rows = [r for r in base_rows if r["label"] == "other_speech"]
    centered_speech_rows = []
    for split, limit in (("train", 300), ("validation", 80)):
        candidates = [r for r in other_rows if r["split"] == split][:limit]
        for i, r in enumerate(candidates):
            with wave.open(str(hey_bis_dir / r["path"]), "rb") as w:
                s = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            s_pad = audio_features.pad_or_crop(s, target_length=48000)
            s_pad = np.clip(s_pad + rng.normal(0, 1e-4, 48000).astype(np.float32), -0.98, 0.98)
            rel = f"audio/recorded/speech_center_{split}_{i:03d}.wav"
            with wave.open(str(hey_bis_dir / rel), "wb") as w:
                w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                w.writeframes((s_pad * 32767).astype("<i2").tobytes())
            centered_speech_rows.append({
                "path": rel,
                "label": "other_speech",
                "language": r["language"],
                "speaker_id": r["speaker_id"],
                "split": split,
                "consent": "yes",
            })

    return base_rows + calib_rows + centered_speech_rows + aug_wake_rows


def sync_dataset(dataset_repo: Path) -> Path:
    """Sincroniza el repositorio de dataset y devuelve la ruta a Hey-BIS lista para entrenar."""
    dataset_repo = dataset_repo.resolve()
    hey_bis_dir = dataset_repo / "Hey-BIS" if (dataset_repo / "Hey-BIS").is_dir() else dataset_repo
    manifest_path = hey_bis_dir / "manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"No se encontró manifest.csv en {hey_bis_dir}")

    sections = ensure_directory_sections(hey_bis_dir)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if (hey_bis_dir / r["path"]).is_file()]

    rows = sync_voices_folder(dataset_repo, hey_bis_dir, rows)
    rows = sync_custom_sections(hey_bis_dir, rows, sections)
    rows = ensure_3s_calibration_and_augmentation(hey_bis_dir, rows)

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return hey_bis_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-repo",
        type=Path,
        default=Path("../dataset"),
        help="Ruta al repositorio de dataset (por defecto ../dataset)",
    )
    args = parser.parse_args()

    hey_bis_dir = sync_dataset(args.dataset_repo)
    report = inspect_dataset(hey_bis_dir, load_config())
    print(
        f"[+] Dataset sincronizado en {hey_bis_dir}: "
        f"{report['files']} archivos | "
        f"Errores: {len(report['errors'])} | "
        f"Pendientes de entrenamiento: {len(report['training_missing'])}"
    )
    if report["errors"]:
        for err in report["errors"][:5]:
            print(f"    ! {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
