import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import main


class DatasetTests(unittest.TestCase):
    def setUp(self):
        # Audio sintetico solo para probar el validador; no es un dataset.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "audio").mkdir()
        self.config = main.load_config()
        self.config["read_block_bytes"] = 128

    def make_wav(self, name="a.wav", rate=16_000):
        path = self.root / "audio" / name
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(b"\0\0" * 1600)
        return path

    def manifest(self, *rows):
        with (self.root / "manifest.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(main.FIELDS)
            writer.writerows(rows)

    def test_counts_real_bytes_duration_and_progress_toward_target(self):
        path = self.make_wav()
        self.manifest(("audio/a.wav", "wake_word", "es", "p01", "train", "yes"))
        report = main.inspect_dataset(self.root, self.config)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["dataset_bytes"], path.stat().st_size)
        self.assertAlmostEqual(report["audio_hours"], 0.1 / 3600)
        self.assertFalse(report["target_reached"])
        self.assertEqual(report["bytes_remaining"], self.config["dataset_target_bytes"] - path.stat().st_size)
        self.assertEqual(report["progress_percent"], round(path.stat().st_size / self.config["dataset_target_bytes"] * 100, 4))
        self.assertEqual(report["missing_labels"], ["background", "other_speech"])

    def test_read_requests_stay_within_block_size(self):
        path = self.make_wav()
        with wave.open(str(path), "rb") as audio:
            with patch.object(audio, "readframes", wraps=audio.readframes) as read:
                with patch("main.wave.open", return_value=audio):
                    main.inspect_wav(path, 128)
                self.assertGreater(read.call_count, 1)
                self.assertTrue(all(call.args[0] * 2 <= 128 for call in read.call_args_list))

    def test_rejects_speaker_shared_with_test(self):
        self.make_wav()
        self.make_wav("b.wav")
        self.manifest(
            ("audio/a.wav", "wake_word", "es", "p01", "train", "yes"),
            ("audio/b.wav", "wake_word", "en", "p01", "test", "yes"),
        )
        report = main.inspect_dataset(self.root, self.config)
        self.assertEqual(report["files"], 1)
        self.assertIn("fuga de datos", report["errors"][0])

    def test_rejects_duplicate_path(self):
        self.make_wav()
        row = ("audio/a.wav", "wake_word", "es", "p01", "train", "yes")
        self.manifest(row, row)
        report = main.inspect_dataset(self.root, self.config)
        self.assertEqual(report["files"], 1)
        self.assertIn("mas de una vez", report["errors"][0])

    def test_rejects_truncated_audio_and_wrong_sample_rate(self):
        truncated = self.make_wav()
        truncated.write_bytes(truncated.read_bytes()[:-10])
        self.make_wav("b.wav", rate=44_100)
        self.manifest(
            ("audio/a.wav", "wake_word", "es", "p01", "train", "yes"),
            ("audio/b.wav", "other_speech", "en", "p02", "test", "yes"),
        )
        report = main.inspect_dataset(self.root, self.config)
        self.assertEqual(report["dataset_bytes"], 0)
        self.assertIn("truncado", report["errors"][0])
        self.assertIn("16000 Hz", report["errors"][1])

    def test_rejects_files_outside_dataset(self):
        path = self.make_wav()
        root = self.root / "nested"
        root.mkdir()
        self.manifest(("../audio/a.wav", "wake_word", "es", "p01", "train", "yes"))
        (self.root / "manifest.csv").rename(root / "manifest.csv")
        self.assertTrue(path.exists())
        report = main.inspect_dataset(root, self.config)
        self.assertEqual(report["files"], 0)
        self.assertIn("dentro", report["errors"][0])

    def test_rejects_missing_consent_and_bad_language(self):
        self.make_wav()
        self.manifest(
            ("audio/a.wav", "wake_word", "es", "p01", "train", "no"),
            ("audio/a.wav", "background", "es", "room01", "train", "yes"),
        )
        report = main.inspect_dataset(self.root, self.config)
        self.assertIn("permiso", report["errors"][0])
        self.assertIn("Idioma", report["errors"][1])

    def test_init_preserves_manifest_and_empty_dataset_fails(self):
        with contextlib.redirect_stdout(io.StringIO()):
            main.initialize(self.root)
        manifest = self.root / "manifest.csv"
        content = manifest.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            main.initialize(self.root)
        self.assertEqual(manifest.read_bytes(), content)
        self.assertTrue(main.inspect_dataset(self.root, self.config)["errors"])

    def test_cli_requires_target_only_when_requested(self):
        self.make_wav()
        self.manifest(("audio/a.wav", "wake_word", "es", "p01", "train", "yes"))
        args = ["main.py", "inspect", "--data-dir", str(self.root)]
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("sys.argv", args):
                self.assertEqual(main.main(), 0)
            with patch("sys.argv", args + ["--require-target"]):
                self.assertEqual(main.main(), 1)

    def test_metadata_does_not_count_toward_audio_target(self):
        self.make_wav()
        self.manifest(("audio/a.wav", "wake_word", "es", "p01", "train", "yes"))
        (self.root / "notes.txt").write_bytes(b"x" * 5000)
        self.config["dataset_target_bytes"] = 6000
        report = main.inspect_dataset(self.root, self.config)
        self.assertLess(report["dataset_bytes"], 6000)
        self.assertGreater(report["storage_bytes"], 6000)
        self.assertFalse(report["target_reached"])
        self.assertEqual(report["bytes_remaining"], 6000 - report["dataset_bytes"])

    def test_target_boundary_and_exceeding_target_are_valid(self):
        size = self.make_wav().stat().st_size
        self.manifest(("audio/a.wav", "wake_word", "es", "p01", "train", "yes"))
        for target in (size - 1, size, size + 1):
            with self.subTest(target=target):
                self.config["dataset_target_bytes"] = target
                report = main.inspect_dataset(self.root, self.config)
                self.assertEqual(report["target_reached"], size >= target)
                self.assertEqual(report["bytes_remaining"], max(0, target - size))
                self.assertEqual(report["errors"], [])


if __name__ == "__main__":
    unittest.main()
