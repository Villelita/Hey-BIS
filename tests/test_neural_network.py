import tempfile
import unittest
from pathlib import Path
import numpy as np

import audio_features
import neural_network
from audio_classifier import AudioNoiseClassifier


class NeuralNetworkTests(unittest.TestCase):
    def setUp(self):
        self.model = neural_network.WakeWordNeuralNetwork(
            input_dim=3136,
            hidden_dim1=32,
            hidden_dim2=16,
            num_classes=3,
        )

    def test_forward_output_shape_and_probability_sum(self):
        batch_size = 5
        x = np.random.randn(batch_size, 3136).astype(np.float32)
        probs = self.model.predict_proba(x)

        self.assertEqual(probs.shape, (batch_size, 3))
        # Cada fila debe sumar 1.0 (propiedad de Softmax)
        row_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(batch_size), atol=1e-5)

    def test_train_step_reduces_loss(self):
        x = np.random.randn(8, 3136).astype(np.float32)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        optimizer = neural_network.AdamOptimizer(self.model.dense_layers, learning_rate=0.05)

        loss_initial = self.model.train_step(x, y, optimizer)
        for _ in range(20):
            loss_final = self.model.train_step(x, y, optimizer)

        self.assertLess(loss_final, loss_initial)

    def test_save_and_load_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            weight_path = Path(temp_dir) / "test_weights.npz"
            self.model.save_weights(weight_path)
            self.assertTrue(weight_path.is_file())

            new_model = neural_network.WakeWordNeuralNetwork(
                input_dim=3136,
                hidden_dim1=32,
                hidden_dim2=16,
                num_classes=3,
            )
            new_model.load_weights(weight_path)

            np.testing.assert_array_equal(self.model.dense1.weights, new_model.dense1.weights)
            np.testing.assert_array_equal(self.model.dense3.bias, new_model.dense3.bias)

    def test_audio_feature_extraction(self):
        # 1 segundo a 16 kHz
        fake_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        features = audio_features.extract_features(fake_audio, n_mels=32)

        self.assertEqual(features.shape, (3136,))
        self.assertFalse(np.isnan(features).any())

    def test_classifier_classify_samples(self):
        classifier = AudioNoiseClassifier()
        fake_audio = np.random.randn(16000).astype(np.float32)
        result = classifier.classify_samples(fake_audio)

        self.assertIn("label", result)
        self.assertIn("is_wake_word", result)
        self.assertIn("probabilities", result)
        self.assertIn(result["label"], neural_network.LABELS)

    def test_summary_spec(self):
        summary = self.model.summary()
        self.assertEqual(summary["input_dimension"], 3136)
        self.assertGreater(summary["total_parameters"], 0)
        self.assertIn("weights_memory_kb", summary)


if __name__ == "__main__":
    unittest.main()
