"""Regresiones de los fallos reales y flujo completo con datos SINTETICOS."""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_classifier import ActivationGate, AudioNoiseClassifier
import audio_features
from evaluate_wake_word import evaluate, event_metrics
from main import FIELDS, inspect_dataset, load_config
import microphone
from neural_network import AdamOptimizer, WakeWordNeuralNetwork, threshold_predictions
from prepare_audio import prepare_dataset
from train_wake_word import load_dataset_split, metrics_from_probabilities, train_network


def write_wav(path, rate=16000, seconds=1, frequency=440):
    samples = np.rint(5000 * np.sin(2 * np.pi * frequency * np.arange(round(rate * seconds)) / rate)).astype('<i2')
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as target:
        target.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        target.writeframes(samples.tobytes())


def manifest(root, rows):
    with (root / 'manifest.csv').open('w', newline='', encoding='utf-8') as target:
        writer = csv.writer(target)
        writer.writerow(FIELDS)
        writer.writerows(rows)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        np.random.seed(123)

    def trained_weights(self):
        model = WakeWordNeuralNetwork(hidden_dim1=8, hidden_dim2=4)
        model.train_step(np.zeros((3, 3136), np.float32), np.arange(3), AdamOptimizer(model.dense_layers))
        path = self.root / 'model.npz'
        model.save_weights(path)
        return path

    def test_threshold_and_metrics_agree_below_threshold(self):
        probs = np.array([[.3, .2, .5], [.1, .2, .7]], np.float32)
        np.testing.assert_array_equal(threshold_predictions(probs, .6), [0, 2])
        model = WakeWordNeuralNetwork()
        with patch.object(model, 'predict_proba', return_value=probs[:1]):
            result = model.classify_audio_samples(np.zeros(16000, np.float32), .6)
        self.assertEqual(result['label'], 'background')
        self.assertFalse(result['is_wake_word'])
        self.assertAlmostEqual(result['confidence'], .3)
        metrics = metrics_from_probabilities(probs, np.array([2, 2]), .6)
        self.assertEqual(metrics['wake_recall'], .5)

    def test_inference_rejects_missing_or_untrained_weights(self):
        with self.assertRaises(ValueError):
            AudioNoiseClassifier()
        with self.assertRaises(FileNotFoundError):
            AudioNoiseClassifier(self.root / 'missing.npz')
        path = self.root / 'random.npz'
        WakeWordNeuralNetwork().save_weights(path)
        with self.assertRaisesRegex(ValueError, 'no ha sido entrenado'):
            AudioNoiseClassifier(path)

    def test_inference_loads_custom_architecture_without_optimizer_buffers(self):
        classifier = AudioNoiseClassifier(self.trained_weights())
        self.assertEqual(classifier.model.hidden_dim1, 8)
        self.assertIsNone(classifier.model.dense1.m_w)
        self.assertIsNone(classifier.model.dense1.grad_weights)

    def test_invalid_serialized_shapes_are_rejected(self):
        path = self.trained_weights()
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        arrays['w1'] = np.zeros((1, 1), np.float32)
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'Pesos invalidos'):
            AudioNoiseClassifier(path)

    def test_windows_cover_tail_and_count_one_event_per_positive_run(self):
        classifier = AudioNoiseClassifier(self.trained_weights())
        with patch.object(classifier, 'classify_samples', return_value={'is_wake_word': True}):
            windows = classifier.scan_streaming_window(np.zeros(19200, np.float32))
        self.assertEqual(windows[-1]['time_end_sec'], 1.2)
        self.assertEqual(sum(w['activation'] for w in windows), 1)
        with self.assertRaises(ValueError):
            list(audio_features.audio_windows(np.zeros(16000), hop_size=0))

    def test_event_gate_requires_release_and_cooldown(self):
        gate = ActivationGate(1)
        self.assertTrue(gate.update(True, 1))
        self.assertFalse(gate.update(True, 3))
        self.assertFalse(gate.update(False, 3.1))
        self.assertTrue(gate.update(True, 3.2))
        gate.update(False, 3.3)
        self.assertFalse(gate.update(True, 3.4))

    def test_false_events_use_actual_duration_not_window_duration(self):
        path = self.root / 'conversation.wav'
        write_wav(path, seconds=2.25)
        classifier = AudioNoiseClassifier(self.trained_weights())
        with patch.object(classifier, 'classify_samples', return_value={'is_wake_word': True}):
            metrics = event_metrics(classifier, [path])
        self.assertEqual(metrics['false_activations'], 1)
        self.assertEqual(metrics['audio_seconds'], 2.25)
        self.assertEqual(metrics['false_activations_per_hour'], 1600)

    def test_conversion_preserves_originals_and_is_idempotent(self):
        path = self.root / 'audio' / 'source.wav'
        write_wav(path, rate=44100)
        original = path.read_bytes()
        manifest(self.root, [('audio/source.wav', 'background', 'none', 'room', 'train', 'yes')])
        with contextlib.redirect_stdout(io.StringIO()):
            result = prepare_dataset(self.root)
            repeated = prepare_dataset(self.root)
        self.assertEqual(result['converted'], 1)
        self.assertEqual(repeated['converted'], 0)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(inspect_dataset(self.root, load_config())['errors'], [])
        self.assertTrue(list((self.root / 'sources' / 'preparation').glob('manifest-*.csv')))

    def test_training_rejects_background_only_without_saving_model(self):
        write_wav(self.root / 'audio/bg.wav')
        manifest(self.root, [('audio/bg.wav', 'background', 'none', 'room', 'train', 'yes')])
        path = self.root / 'output.npz'
        with self.assertRaisesRegex(ValueError, 'Faltan grabaciones'):
            train_network(self.root, epochs=1, save_path=path)
        self.assertFalse(path.exists())

    def test_loader_rejects_wrong_rate_instead_of_silently_skipping(self):
        write_wav(self.root / 'audio/bg.wav', rate=44100)
        manifest(self.root, [('audio/bg.wav', 'background', 'none', 'room', 'train', 'yes')])
        with self.assertRaisesRegex(ValueError, 'Formato no compatible'):
            load_dataset_split(self.root, 'train')

    def test_copied_content_cannot_cross_partitions_with_different_speaker_ids(self):
        write_wav(self.root / 'audio/a.wav')
        (self.root / 'audio/b.wav').write_bytes((self.root / 'audio/a.wav').read_bytes())
        manifest(self.root, [('audio/a.wav', 'other_speech', 'es', 'p01', 'train', 'yes'),
                             ('audio/b.wav', 'other_speech', 'es', 'p02', 'test', 'yes')])
        report = inspect_dataset(self.root, load_config())
        self.assertTrue(any('contenido' in error for error in report['errors']))

    def test_negative_audio_is_segmented_and_positive_never_silently_cropped(self):
        write_wav(self.root / 'audio/long.wav', seconds=2.25)
        manifest(self.root, [('audio/long.wav', 'other_speech', 'es', 'p01', 'train', 'yes')])
        x, _, _ = load_dataset_split(self.root, 'train')
        self.assertEqual(len(x), 3)
        manifest(self.root, [('audio/long.wav', 'wake_word', 'es', 'p01', 'train', 'yes')])
        with self.assertRaisesRegex(ValueError, 'hasta 1 segundo'):
            load_dataset_split(self.root, 'train')

    def make_bilingual_fixture(self):
        rows = []
        for si, split in enumerate(('train', 'validation', 'test')):
            for li, label in enumerate(('background', 'other_speech', 'wake_word')):
                for lang in (('none',) if label == 'background' else ('es', 'en')):
                    name = f'audio/{split}_{label}_{lang}.wav'
                    write_wav(self.root / name, frequency=200 + 500 * li + 13 * si + (lang == 'en') * 17)
                    rows.append((name, label, lang, f'p{si}_{li}_{lang}', split, 'yes'))
        manifest(self.root, rows)

    def test_synthetic_end_to_end_training_evaluation_and_manifest_guard(self):
        self.make_bilingual_fixture()
        path = self.root / 'trained.npz'
        with contextlib.redirect_stdout(io.StringIO()):
            report = train_network(self.root, epochs=2, save_path=path)
            evaluated = evaluate(self.root, path, self.root / 'evaluation.json')
        self.assertFalse(report['test_evaluated'])
        self.assertEqual(len(report['history']), 2)
        self.assertTrue(path.with_suffix('.training.json').exists())
        self.assertEqual(evaluated['clips_and_windows']['total_samples'], 5)
        self.assertIsNone(evaluated['ordinary_talk'])
        ordinary = self.root / 'ordinary.wav'
        write_wav(ordinary, seconds=2.25, frequency=333)
        with contextlib.redirect_stdout(io.StringIO()):
            with_ordinary = evaluate(self.root, path, self.root / 'ordinary.json', [ordinary])
        self.assertEqual(with_ordinary['ordinary_talk']['audio_seconds'], 2.25)
        with self.assertRaisesRegex(ValueError, 'train/validation'):
            evaluate(self.root, path, self.root / 'bad.json', [self.root / 'audio/train_other_speech_es.wav'])
        with (self.root / 'manifest.csv').open('a') as target:
            target.write('\n')
        with self.assertRaisesRegex(ValueError, 'manifiesto cambio'):
            evaluate(self.root, path, self.root / 'bad.json')

    def test_evaluation_rejects_modified_audio_after_training(self):
        self.make_bilingual_fixture()
        path = self.root / 'trained.npz'
        with contextlib.redirect_stdout(io.StringIO()):
            train_network(self.root, epochs=1, save_path=path)
        write_wav(self.root / 'audio/test_wake_word_es.wav', frequency=1723)
        with self.assertRaisesRegex(ValueError, 'audios cambio'):
            evaluate(self.root, path, self.root / 'bad.json')

    def test_recording_requires_consent_before_accessing_microphone(self):
        args = argparse.Namespace(consent=False)
        with patch('microphone.sounddevice_module') as sd:
            with self.assertRaisesRegex(ValueError, 'permiso'):
                microphone.record(args)
            sd.assert_not_called()

    def test_recording_registers_pcm_and_blocks_speaker_leak(self):
        args = argparse.Namespace(consent=True, data_dir=self.root, label='wake_word', language='es',
                                  speaker='p01', split='train', seconds=1.0, device=None)
        sd = Mock()
        sd.rec.return_value = np.zeros((16000, 1), dtype=np.int16)
        sd.get_status.return_value = False
        with patch('microphone.sounddevice_module', return_value=sd), contextlib.redirect_stdout(io.StringIO()):
            microphone.record(args)
        report = inspect_dataset(self.root, load_config())
        self.assertEqual(report['files'], 1)
        self.assertEqual(report['errors'], [])
        args.split = 'test'
        with patch('microphone.sounddevice_module') as sd:
            with self.assertRaisesRegex(ValueError, 'otra particion'):
                microphone.record(args)
            sd.assert_not_called()

    def test_live_capture_processes_callback_blocks_and_logs(self):
        args = argparse.Namespace(weights=self.trained_weights(), threshold=None, cooldown=1,
                                  hop_size=4000, seconds=.05, device=None, log=self.root / 'live.jsonl')
        sd = Mock()
        stream = Mock()
        def input_stream(**kwargs):
            def enter():
                for _ in range(5):
                    kwargs['callback'](np.zeros((4000, 1), np.float32), 4000, None, False)
            stream.__enter__ = Mock(side_effect=enter)
            stream.__exit__ = Mock(return_value=False)
            return stream
        sd.InputStream.side_effect = input_stream
        with patch('microphone.sounddevice_module', return_value=sd), contextlib.redirect_stdout(io.StringIO()):
            microphone.listen(args)
        logged = [json.loads(line) for line in args.log.read_text().splitlines()]
        self.assertEqual(logged[0]['type'], 'session')
        self.assertGreater(logged[-1]['windows'], 0)


if __name__ == '__main__':
    unittest.main()
