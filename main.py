"""Etapa 1 de Hey BIS: preparar y revisar datos, sin entrenar todavia."""

import argparse
import csv
import json
from pathlib import Path
import wave


PROJECT_DIR = Path(__file__).resolve().parent
FIELDS = ("path", "label", "language", "speaker_id", "split", "consent")
LABELS = {"wake_word", "other_speech", "background"}
SPLITS = {"train", "validation", "test"}


def load_config():
    with (PROJECT_DIR / "config.json").open(encoding="utf-8") as source:
        config = json.load(source)
    for key in ("dataset_target_bytes", "pi_ram_bytes", "read_block_bytes"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} debe ser un entero positivo.")
    return config


def initialize(data_dir):
    (data_dir / "audio").mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "manifest.csv"
    try:
        with manifest.open("x", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow(FIELDS)
    except FileExistsError:
        pass
    print(f"Carpeta de datos: {data_dir}")
    print("Listo. Agrega grabaciones reales y sus filas en manifest.csv.")


def inspect_wav(path, block_bytes):
    """Comprueba PCM mono 16 kHz/16 bits leyendo solo un bloque a la vez."""
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                audio.getcomptype()) != (1, 2, 16_000, "NONE"):
            raise ValueError("Se requiere WAV PCM, mono, 16000 Hz y 16 bits.")
        expected_bytes = audio.getnframes() * 2
        total_bytes = 0
        frames_per_block = max(1, block_bytes // 2)
        while True:
            block = audio.readframes(frames_per_block)
            if not block:
                break
            total_bytes += len(block)
            # Liberar antes de pedir el siguiente bloque evita retener dos.
            del block
        if total_bytes != expected_bytes:
            raise ValueError("Audio truncado: faltan muestras declaradas en el WAV.")
        if total_bytes == 0:
            raise ValueError("La grabacion esta vacia.")
    return total_bytes / 32_000


def inspect_dataset(data_dir, config):
    root = data_dir.resolve(strict=True)
    seen_paths = set()
    speaker_splits = {}
    counts = {split: {label: 0 for label in sorted(LABELS)} for split in sorted(SPLITS)}
    total_bytes = 0
    seconds = 0.0
    errors = []
    with (root / "manifest.csv").open(newline="", encoding="utf-8-sig") as source:
        rows = csv.DictReader(source)
        if rows.fieldnames != list(FIELDS):
            raise ValueError(f"Columnas requeridas, en este orden: {','.join(FIELDS)}")
        for line, row in enumerate(rows, start=2):
            try:
                if None in row or any(not row.get(field, "") for field in FIELDS):
                    raise ValueError("Fila incompleta o con columnas adicionales.")
                label, language, speaker, split, consent = (
                    row[key] for key in FIELDS[1:]
                )
                if label not in LABELS or split not in SPLITS:
                    raise ValueError("Etiqueta o particion desconocida.")
                allowed_languages = {"none"} if label == "background" else {"es", "en"}
                if language not in allowed_languages:
                    raise ValueError("Idioma: es/en para voz; none para fondo.")
                if consent not in {"yes", "licensed"}:
                    raise ValueError("Se requiere permiso (yes) o una fuente licenciada (licensed).")
                relative = Path(row["path"])
                if relative.is_absolute():
                    raise ValueError("Usa una ruta relativa a la carpeta de datos.")
                path = (root / relative).resolve(strict=True)
                if not path.is_relative_to(root) or path.suffix.lower() != ".wav":
                    raise ValueError("El WAV debe estar dentro de la carpeta de datos.")
                if consent == "licensed" and (
                    label != "background"
                    or not path.is_relative_to(root / "audio" / "esc50")
                    or not (root / "sources" / "esc50" / "LICENSE").is_file()
                ):
                    raise ValueError("licensed requiere audio de fondo ESC-50 y su archivo LICENSE.")
                if path in seen_paths:
                    raise ValueError("El mismo archivo aparece mas de una vez.")
                if speaker in speaker_splits and speaker_splits[speaker] != split:
                    raise ValueError(f"{speaker} aparece en varias particiones; hay fuga de datos.")
                duration = inspect_wav(path, config["read_block_bytes"])
                seen_paths.add(path)
                speaker_splits[speaker] = split
                total_bytes += path.stat().st_size
                seconds += duration
                counts[split][label] += 1
            except (OSError, ValueError, wave.Error, EOFError) as error:
                errors.append(f"Fila {line}: {error}")
    if not seen_paths:
        errors.append("No hay grabaciones validas en el manifiesto.")
    storage_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    target_bytes = config["dataset_target_bytes"]
    missing_labels = sorted(label for label in LABELS
                            if not any(counts[split][label] for split in SPLITS))
    return {
        "files": len(seen_paths),
        "dataset_bytes": total_bytes,
        "storage_bytes": storage_bytes,
        "dataset_target_bytes": target_bytes,
        "target_reached": total_bytes >= target_bytes,
        "bytes_remaining": max(0, target_bytes - total_bytes),
        "progress_percent": round(total_bytes / target_bytes * 100, 4),
        "audio_hours": seconds / 3600,
        "counts": counts,
        "missing_labels": missing_labels,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Hey BIS - etapa 1: datos de audio")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("plan", help="Mostrar arquitectura y objetivos")
    for command in ("init", "inspect"):
        subparser = subcommands.add_parser(command)
        subparser.add_argument("--data-dir", type=Path, required=True,
                               help="Carpeta de la Raspberry montada en la computadora")
        if command == "inspect":
            subparser.add_argument("--require-target", action="store_true",
                                   help="Fallar si aun no se alcanza la meta de datos")
    args = parser.parse_args()
    try:
        config = load_config()
        if args.command == "plan":
            print("Proyecto 1: detectar Hey BIS en espanol e ingles.")
            print("Raspberry Pi: guardar audio y manifest.csv.")
            print("Computadora: validar, preparar, entrenar y evaluar.")
            print(f"Meta de audio para almacenar en la Pi: {config['dataset_target_bytes']:,} bytes.")
            print(f"RAM de la Pi: {config['pi_ram_bytes']:,} bytes.")
            print("La meta de datos no es un limite de almacenamiento. No hay que llenar la RAM.")
            print(f"Bloque de lectura en la computadora: {config['read_block_bytes']:,} bytes.")
            print("La conexion y los recursos reales de la Pi siguen pendientes de comprobar.")
        elif args.command == "init":
            initialize(args.data_dir)
        else:
            report = inspect_dataset(args.data_dir, config)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            if report["errors"] or (args.require_target and not report["target_reached"]):
                return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
