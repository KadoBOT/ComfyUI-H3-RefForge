"""Audio backend regressions without loading codec weights."""
import builtins
import contextlib
import importlib
import math
import sys
import types
import unittest
from unittest.mock import Mock, patch

import test_loaders as loaders

torch = loaders.torch
AUDIO = importlib.import_module(loaders.N.__package__ + ".audio")


class AudioResampleTests(unittest.TestCase):
    def setUp(self):
        self.pieces = []
        pieces = self.pieces

        class Codec:
            def encode(self, waveform):
                pieces.append(waveform.clone())
                return torch.zeros(1, 32, 2, math.ceil(waveform.shape[-1] / 800))

        self.vae = types.SimpleNamespace(first_stage_model=Codec(), patcher=object(),
                                         device="cpu", vae_dtype=torch.float32)
        for mocked in (patch.object(AUDIO, "MiniMaxH3AudioVAE", Codec),
                       patch.object(AUDIO.model_management, "load_models_gpu"),
                       patch.object(AUDIO.model_management, "cuda_device_context",
                                    side_effect=lambda device: contextlib.nullcontext())):
            mocked.start()
            self.addCleanup(mocked.stop)

    def backend(self):
        def resample(waveform, orig_freq, new_freq):
            length = math.ceil(waveform.shape[-1] * new_freq / orig_freq)
            return torch.ones(*waveform.shape[:-1], length)
        return types.SimpleNamespace(resample=Mock(side_effect=resample))

    def test_native_backend_without_torchaudio_clips_before_resampling(self):
        for rate, channels in ((44100, 1), (48000, 2)):
            with self.subTest(rate=rate, channels=channels):
                native = self.backend()
                self.pieces.clear()
                with patch.dict(sys.modules, {"comfy.audio": native, "torchaudio": None,
                                              "torchaudio.functional": None}):
                    z = AUDIO.encode_audio(self.vae, {
                        "waveform": torch.zeros(1,channels,rate*2), "sample_rate": rate},
                        max_seconds=1.25, chunk_seconds=1)
                wave, orig, new = native.resample.call_args.args
                self.assertEqual(tuple(wave.shape), (1,2,round(rate*1.25)))
                self.assertEqual((orig,new), (rate,32000))
                self.assertEqual([p.shape[-1] for p in self.pieces], [32000,8000])
                self.assertEqual(tuple(z.shape), (1,32,2,50))

    def test_older_comfyui_uses_torchaudio(self):
        legacy = self.backend()
        with patch.dict(sys.modules, {"comfy.audio": None, "torchaudio.functional": legacy}):
            AUDIO.encode_audio(self.vae, {"waveform": torch.zeros(1,2,4800), "sample_rate": 48000})
        legacy.resample.assert_called_once()
        self.assertEqual(self.pieces[0].shape[-1], 3200)

    def test_32000_needs_neither_resampler(self):
        with patch.dict(sys.modules, {"comfy.audio": None, "torchaudio": None,
                                      "torchaudio.functional": None}):
            AUDIO.encode_audio(self.vae, {"waveform": torch.ones(1,1,3200), "sample_rate": 32000})
        torch.testing.assert_close(self.pieces[0], torch.ones(1,2,3200))

    def test_native_dependency_error_is_not_hidden_by_fallback(self):
        original_import = builtins.__import__
        def failing_import(name, *args, **kwargs):
            if name == "comfy.audio":
                raise ModuleNotFoundError("Missing native dependency", name="scipy")
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=failing_import):
            with self.assertRaises(ModuleNotFoundError) as error:
                AUDIO.encode_audio(self.vae, {"waveform": torch.zeros(1,2,4800), "sample_rate": 48000})
        self.assertEqual(error.exception.name, "scipy")


if __name__ == "__main__":
    unittest.main()
