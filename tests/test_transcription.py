"""Verify language metadata and silence handling without downloading a model."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import speech_transcription


class TestTranscription(unittest.TestCase):
    def test_reply_locale_follows_detected_audio_language(self):
        for language, locale, text in (
            ("en", "en-US", "What time is it?"),
            ("es", "es-MX", "¿Qué hora es?"),
            ("en", "es-MX", "Hola Bis."),
        ):
            with self.subTest(language=language):
                model = Mock()
                model.transcribe.return_value = (
                    iter([SimpleNamespace(text=text)]),
                    SimpleNamespace(language=language),
                )
                with patch.object(speech_transcription, "MODEL", model):
                    result = speech_transcription.transcribe_samples(np.full(48000, 0.1, np.float32))
                self.assertEqual(result, {"text": text, "language": locale})
                self.assertEqual(model.transcribe.call_args.kwargs["task"], "transcribe")

    def test_silence_does_not_invent_words(self):
        model = Mock()
        with patch.object(speech_transcription, "MODEL", model):
            result = speech_transcription.transcribe_samples(np.zeros(48000, np.float32))
        self.assertEqual(result["text"], "")
        model.transcribe.assert_not_called()

    def test_nonfinite_audio_is_rejected(self):
        with self.assertRaises(ValueError):
            speech_transcription.transcribe_samples(np.array([np.nan]))


if __name__ == "__main__":
    unittest.main()
