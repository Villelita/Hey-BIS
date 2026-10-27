"""Importar ESC-50 como fondo, conservando fuentes, particiones y atribuciones."""

import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import wave

from main import FIELDS, initialize


def write_csv(path, fields, rows):
    """Reemplaza el CSV solo cuando se termino de escribir su nueva version."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, newline="",
                                     encoding="utf-8", delete=False) as output:
        temporary = Path(output.name)
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def convert_clip(path):
    import numpy as np
    from scipy.signal import resample_poly

    # ESC-50 contiene clips de 5 s. Solo se mantiene un clip en memoria.
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(),
                source.getnframes(), source.getcomptype()) != (1, 2, 44_100, 220_500, "NONE"):
            raise ValueError(f"Formato inesperado en ESC-50: {path.name}")
        raw = source.readframes(220_501)
        if len(raw) != 441_000:
            raise ValueError(f"Audio truncado: {path.name}")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    # 44100 * 160 / 441 = 16000 Hz; el filtro evita aliasing al reducir la tasa.
    resampled = resample_poly(samples, 160, 441)
    pcm = np.clip(np.rint(resampled * 32768), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(pcm.tobytes())
    return buffer.getvalue()


def import_dataset(source_dir, data_dir):
    source_dir = source_dir.resolve(strict=True)
    data_dir = data_dir.resolve()
    # Acepta tanto la carpeta del dataset como el contenedor del ZIP extraido.
    if not (source_dir / "meta" / "esc50.csv").is_file():
        source_dir = (source_dir / "ESC-50-master").resolve(strict=True)
    metadata = source_dir / "meta" / "esc50.csv"
    license_path = source_dir / "LICENSE"
    if not license_path.is_file():
        raise ValueError("Falta LICENSE con las atribuciones de ESC-50.")
    with metadata.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("El CSV de ESC-50 esta vacio.")
    group_folds = {}
    filenames = set()
    for row in rows:
        filename = row["filename"]
        fold = int(row["fold"])
        group = row["src_file"]
        if (Path(filename).name != filename or not filename.endswith(".wav")
                or filename in filenames or fold not in range(1, 6) or not group.isdigit()):
            raise ValueError(f"Metadatos invalidos o repetidos: {filename}")
        audio = (source_dir / "audio" / filename).resolve(strict=True)
        if not audio.is_relative_to(source_dir / "audio"):
            raise ValueError(f"El audio sale de la carpeta fuente: {filename}")
        filenames.add(filename)
        # Si una fuente cruza folds, todos sus fragmentos usan el menor fold.
        group_folds[group] = min(group_folds.get(group, fold), fold)
    initialize(data_dir)
    manifest_path = data_dir / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("El manifiesto existente tiene columnas incompatibles.")
        existing = list(reader)
    existing_by_path = {}
    for row in existing:
        key = (data_dir / row["path"]).resolve()
        if key in existing_by_path:
            raise ValueError("El manifiesto existente ya contiene rutas duplicadas.")
        existing_by_path[key] = row
    output_dir = data_dir / "audio" / "esc50"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.resolve().is_relative_to(data_dir):
        raise ValueError("La carpeta de destino sale de data-dir.")
    provenance = []
    imported = 0
    reassigned = 0
    for index, row in enumerate(rows, start=1):
        fold = group_folds[row["src_file"]]
        split = "train" if fold <= 3 else "validation" if fold == 4 else "test"
        original_split = "train" if int(row["fold"]) <= 3 else (
            "validation" if row["fold"] == "4" else "test")
        reassigned += split != original_split
        destination = output_dir / row["filename"]
        item = dict(zip(FIELDS, (
            destination.relative_to(data_dir).as_posix(), "background", "none",
            f"esc50_{row['src_file']}", split, "licensed",
        )))
        previous = existing_by_path.get(destination.resolve())
        if previous is not None and previous != item:
            raise ValueError(f"Hay etiquetas previas diferentes para {destination.name}.")
        encoded = convert_clip(source_dir / "audio" / row["filename"])
        if destination.exists():
            if destination.read_bytes() != encoded:
                raise ValueError(f"No se sobrescribira un audio diferente: {destination}")
        else:
            with destination.open("xb") as output:
                output.write(encoded)
        if previous is None:
            existing.append(item)
            imported += 1
        provenance.append({
            "path": item["path"], "original_filename": row["filename"],
            "category": row["category"], "src_file": row["src_file"],
            "original_fold": row["fold"], "split": split,
        })
        if index % 200 == 0:
            print(f"Convertidos {index}/{len(rows)}")
    source_output = data_dir / "sources" / "esc50"
    source_output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(license_path, source_output / "LICENSE")
    shutil.copyfile(metadata, source_output / "esc50.csv")
    write_csv(source_output / "provenance.csv", tuple(provenance[0]), provenance)
    (source_output / "ATTRIBUTION.md").write_text(
        "# ESC-50\n\nKarol J. Piczak, ESC: Dataset for Environmental Sound Classification, 2015.\n"
        "Fuente: https://github.com/karolpiczak/ESC-50\n\n"
        "Ver LICENSE para las condiciones del dataset y atribuciones por clip.\n"
        "Cambios: remuestreo de 44100 a 16000 Hz (PCM mono de 16 bits);\n"
        "etiqueta background; particiones agrupadas por src_file.\n"
        "No son grabaciones propias ni ejemplos de la frase Hey BIS.\n",
        encoding="utf-8",
    )
    write_csv(manifest_path, FIELDS, existing)
    return {
        "source_files": len(rows), "new_manifest_rows": imported,
        "audio_bytes": len(rows) * 160_044,
        "split_counts": dict(Counter(row["split"] for row in provenance)),
        "clips_reassigned_to_keep_source_together": reassigned,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = import_dataset(args.source_dir, args.data_dir)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, wave.Error, EOFError, ImportError) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
