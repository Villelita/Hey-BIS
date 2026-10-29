import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import import_esc50
import main


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.destination = self.root / "data"
        (self.source / "audio").mkdir(parents=True)
        (self.source / "meta").mkdir()
        (self.source / "LICENSE").write_text("TEST FIXTURE - not real ESC-50", encoding="utf-8")
        # Una fuente en folds 4 y 5 para comprobar que no cruce validation/test.
        self.rows = [
            ("4-123-A-0.wav", 4, 0, "dog", "True", "123", "A"),
            ("5-123-B-0.wav", 5, 0, "dog", "True", "123", "B"),
            ("1-456-A-0.wav", 1, 0, "dog", "True", "456", "A"),
        ]
        self.samples = (10000 * np.sin(2 * np.pi * 1000 * np.arange(220500) / 44100)).astype("<i2")
        for row in self.rows:
            with wave.open(str(self.source / "audio" / row[0]), "wb") as output:
                output.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
                output.writeframes(self.samples.tobytes())
        with (self.source / "meta" / "esc50.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(("filename", "fold", "target", "category", "esc10", "src_file", "take"))
            writer.writerows(self.rows)

    def run_import(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return import_esc50.import_dataset(self.source, self.destination)

    def test_import_converts_audio_preserves_sources_and_groups(self):
        original = (self.source / "audio" / self.rows[0][0]).read_bytes()
        result = self.run_import()
        self.assertEqual(result["split_counts"], {"validation": 2, "train": 1})
        self.assertEqual(result["clips_reassigned_to_keep_source_together"], 1)
        report = main.inspect_dataset(self.destination, main.load_config())
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["files"], 3)
        self.assertAlmostEqual(report["audio_hours"], 15 / 3600)
        self.assertEqual((self.source / "audio" / self.rows[0][0]).read_bytes(), original)
        self.assertTrue((self.destination / "sources" / "esc50" / "LICENSE").exists())
        self.assertTrue((self.destination / "sources" / "esc50" / "provenance.csv").exists())
        with wave.open(str(self.destination / "audio" / "esc50" / self.rows[0][0]), "rb") as audio:
            pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
            self.assertEqual(audio.getframerate(), 16000)
            self.assertEqual(len(pcm), 80000)
        # Confirma que el tono no se perdio ni cambio de frecuencia al convertir.
        spectrum = np.abs(np.fft.rfft(pcm.astype(float)))
        frequency = np.fft.rfftfreq(len(pcm), 1 / 16000)[np.argmax(spectrum)]
        self.assertAlmostEqual(frequency, 1000, delta=1)
        self.assertGreater(np.sqrt(np.mean(pcm.astype(float) ** 2)), 6000)

    def test_reimport_is_idempotent_and_preserves_user_recordings(self):
        self.run_import()
        user_audio = self.destination / "audio" / "own.wav"
        with wave.open(str(user_audio), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(b"\0\0" * 1600)
        with (self.destination / "manifest.csv").open("a", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow(("audio/own.wav", "wake_word", "es", "p01", "train", "yes"))
        result = self.run_import()
        self.assertEqual(result["new_manifest_rows"], 0)
        self.assertEqual(main.inspect_dataset(self.destination, main.load_config())["files"], 4)

    def test_refuses_to_overwrite_different_audio(self):
        self.run_import()
        path = self.destination / "audio" / "esc50" / self.rows[0][0]
        path.write_bytes(b"existing user data")
        with self.assertRaisesRegex(ValueError, "sobrescribira"):
            self.run_import()
        self.assertEqual(path.read_bytes(), b"existing user data")


if __name__ == "__main__":
    unittest.main()
