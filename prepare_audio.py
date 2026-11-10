"""Crear copias PCM 16 kHz y actualizar el manifiesto sin tocar originales."""

import argparse
import csv
import hashlib
import io
import json
from math import gcd
from pathlib import Path
import wave

import numpy as np
from scipy.signal import resample_poly

from import_esc50 import write_csv
from main import FIELDS, inspect_wav


def prepare_dataset(data_dir: Path) -> dict:
    root = data_dir.resolve(strict=True)
    manifest = root / "manifest.csv"
    original = manifest.read_bytes()
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Columnas de manifiesto incompatibles.")
        rows = list(reader)
    conversions = []
    seen = set()
    for row in rows:
        if None in row or any(not row.get(k) for k in FIELDS):
            raise ValueError("Fila incompleta en el manifiesto.")
        relative = Path(row["path"])
        path = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not path.is_relative_to(root) or path.suffix.lower() != ".wav":
            raise ValueError(f"Ruta de audio invalida: {relative}")
        if path in seen:
            raise ValueError(f"Audio repetido: {relative}")
        seen.add(path)
        with wave.open(str(path), "rb") as source:
            channels, width, rate, frames, compression, _ = source.getparams()
            if (channels, width, compression) != (1, 2, "NONE") or rate <= 0:
                raise ValueError(f"Se requiere WAV PCM mono de 16 bits: {relative}")
            if rate == 16000:
                inspect_wav(path, 16_000_000)
                continue
            # Un archivo a la vez. Las grabaciones largas deben segmentarse antes.
            if frames > rate * 300:
                raise ValueError(f"Segmenta el audio en clips de hasta 5 minutos: {relative}")
            raw = source.readframes(frames)
        if not frames or len(raw) != frames * 2:
            raise ValueError(f"Audio vacio o truncado: {relative}")
        divisor = gcd(rate, 16000)
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        converted = resample_poly(samples, 16000 // divisor, rate // divisor)
        pcm = np.clip(np.rint(converted), -32768, 32767).astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as target:
            target.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            target.writeframes(pcm.tobytes())
        destination = path.parent / "pcm16k" / path.name
        destination.parent.mkdir(exist_ok=True)
        if not destination.resolve().is_relative_to(root):
            raise ValueError("El destino sale del dataset.")
        encoded = buffer.getvalue()
        if destination.exists():
            if destination.read_bytes() != encoded:
                raise ValueError(f"No se sobrescribira un archivo distinto: {destination}")
        else:
            with destination.open("xb") as target:
                target.write(encoded)
        inspect_wav(destination, 16_000_000)
        new_path = destination.relative_to(root).as_posix()
        conversions.append({"original_path": row["path"], "path": new_path,
                            "original_rate": rate, "sample_rate": 16000,
                            "original_pcm_sha256": hashlib.sha256(raw).hexdigest()})
        row["path"] = new_path
        if len(conversions) % 200 == 0:
            print(f"Convertidos {len(conversions)} audios")
    if conversions:
        # Nunca reemplazar el manifiesto hasta que todas las conversiones concluyan.
        backup = root / "sources" / "preparation"
        backup.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(original).hexdigest()[:16]
        backup_file = backup / f"manifest-{digest}.csv"
        if not backup_file.exists():
            with backup_file.open("xb") as target:
                target.write(original)
        write_csv(backup / f"conversions-{digest}.csv", tuple(conversions[0]), conversions)
        write_csv(manifest, FIELDS, rows)
    return {"converted": len(conversions), "registered_files": len(rows),
            "originals_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        print(json.dumps(prepare_dataset(args.data_dir), indent=2))
    except (OSError, ValueError, KeyError, wave.Error, EOFError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
