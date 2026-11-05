"""Evaluacion final de clips y eventos por hora, con la politica de la demo."""

import argparse
import hashlib
import json
from pathlib import Path

from audio_classifier import AudioNoiseClassifier
import audio_features
from train_wake_word import (evaluate_model, load_dataset_split, read_rows,
                             validate_training_dataset)


def event_metrics(classifier, paths, hop_size=4000, cooldown=1.0):
    seconds = 0.0
    events = []
    for path in paths:
        samples = audio_features.load_wav(path)
        windows = classifier.scan_streaming_window(samples, hop_size=hop_size, cooldown=cooldown)
        seconds += len(samples) / audio_features.SAMPLE_RATE
        events.extend({"path": str(path), **r} for r in windows if r["activation"])
    return {"files": len(paths), "audio_seconds": seconds, "false_activations": len(events),
            "false_activations_per_hour": len(events) * 3600 / seconds if seconds else None,
            "events": events}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(data_dir: Path, weights: Path, output: Path,
             ordinary_talk=None, hop_size=4000, cooldown=1.0):
    dataset_report = validate_training_dataset(data_dir)
    classifier = AudioNoiseClassifier(weights)
    manifest_hash = hashlib.sha256((data_dir / "manifest.csv").read_bytes()).hexdigest()
    if classifier.model.metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("El manifiesto cambio desde el entrenamiento. Usa el dataset registrado con el modelo.")
    if classifier.model.metadata.get("dataset_sha256") != dataset_report["dataset_sha256"]:
        raise ValueError("El contenido de los audios cambio desde el entrenamiento.")
    x, y, _ = load_dataset_split(data_dir, "test")
    report = {"schema_version": 1, "split": "test", "manifest_sha256": manifest_hash,
              "dataset_sha256": dataset_report["dataset_sha256"],
              "model_sha256": file_hash(weights), "threshold": classifier.wake_threshold,
              "window_seconds": audio_features.WINDOW_SECONDS,
              "window_samples": audio_features.WINDOW_SAMPLES,
              "hop_samples": hop_size, "cooldown_seconds": cooldown,
              "clips_and_windows": evaluate_model(classifier.model, x, y), "wake_by_language": {}}
    rows = read_rows(data_dir, "test")
    for language in ("es", "en"):
        positives = [r for r in rows if r["label"] == "wake_word" and r["language"] == language]
        caught = sum(classifier.classify_file(data_dir / r["path"])["is_wake_word"] for r in positives)
        report["wake_by_language"][language] = {"positive_clips": len(positives),
            "detected": caught, "recall": caught / len(positives) if positives else None}
    for label in ("background", "other_speech"):
        paths = [data_dir / r["path"] for r in rows if r["label"] == label]
        report[f"{label}_test_events"] = event_metrics(classifier, paths, hop_size, cooldown)
    if ordinary_talk:
        ordinary_talk = [Path(p).resolve(strict=True) for p in ordinary_talk]
        if len(set(ordinary_talk)) != len(ordinary_talk):
            raise ValueError("Conversacion repetida en --ordinary-talk.")
        # Impide medir como conversacion reservada un archivo usado para ajustar el modelo.
        fitting = read_rows(data_dir, "train") + read_rows(data_dir, "validation")
        fitting_hashes = {file_hash(data_dir / r["path"]) for r in fitting}
        hashes = [file_hash(p) for p in ordinary_talk]
        if len(set(hashes)) != len(hashes) or fitting_hashes.intersection(hashes):
            raise ValueError("La conversacion esta duplicada o fue usada en train/validation.")
        report["ordinary_talk"] = event_metrics(classifier, ordinary_talk, hop_size, cooldown)
        report["ordinary_talk"]["confirmed_without_wake_word"] = True
    else:
        report["ordinary_talk"] = None
        report["pending"] = "Falta conversacion normal reservada para medir la metrica del PDF."
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/evaluation.json"))
    parser.add_argument("--ordinary-talk", nargs="+", type=Path,
                        help="WAV de conversacion reservada que no contenga Hey BIS")
    parser.add_argument("--confirm-no-wake-word", action="store_true")
    parser.add_argument("--hop-size", type=int, default=4000)
    parser.add_argument("--cooldown", type=float, default=1.0)
    args = parser.parse_args()
    if args.ordinary_talk and not args.confirm_no_wake_word:
        parser.error("Confirma que la conversacion no contiene Hey BIS con --confirm-no-wake-word.")
    try:
        report = evaluate(args.data_dir, args.weights, args.output, args.ordinary_talk,
                          args.hop_size, args.cooldown)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError, EOFError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
