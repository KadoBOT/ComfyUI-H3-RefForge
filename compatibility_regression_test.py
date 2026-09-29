"""CPU/MPS preprocessing regressions, without model weights.

Run with ComfyUI's Python from this directory:
    PYTORCH_ENABLE_MPS_FALLBACK=0 python -m unittest compatibility_regression_test -v
MPS tests skip when no Apple GPU is available. For a standalone checkout, add
your ComfyUI directory to PYTHONPATH so common.py can import comfy.utils.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F

from common import sample_video_for_vae, snap_to_causal_grid
from core import _blur_latent, pool_latent
from gauntlet_harness import synchronize


class VideoGridTests(unittest.TestCase):
    def test_h3_grid_and_image_special_case(self):
        for n, expected in ((1,1),(2,1),(4,1),(5,5),(16,5),(17,5),(21,5),
                            (22,22),(23,22),(38,22),(39,39),(101,90),(124,124)):
            with self.subTest(frames=n):
                self.assertEqual(snap_to_causal_grid(n), expected)

    def test_aligned_clips_preserve_every_frame(self):
        for n in (1,5,22,39,124):
            clip = torch.arange(n).reshape(n,1,1,1)
            self.assertIs(sample_video_for_vae(clip), clip)
            self.assertIs(sample_video_for_vae(clip, n), clip)
            self.assertIs(sample_video_for_vae(clip, n+10), clip)

    def test_frame_caps_keep_order_and_endpoints(self):
        for n in (2,4,16,21,22,38,39,101):
            clip = torch.arange(n).reshape(n,1,1,1)
            for limit in (1,4,5,16,22,39,101):
                with self.subTest(frames=n, limit=limit):
                    sampled = sample_video_for_vae(clip, limit).flatten()
                    self.assertLessEqual(len(sampled), min(n, limit))
                    self.assertEqual(sampled[0], 0)
                    if len(sampled) > 1:
                        self.assertEqual(len(sampled) % 17, 5)
                        self.assertEqual(sampled[-1], n-1)
                        self.assertTrue(torch.all(sampled[1:] > sampled[:-1]))

    def test_empty_input_and_invalid_cap_fail_before_encoding(self):
        with self.assertRaises(ValueError):
            sample_video_for_vae(torch.empty(0,1,1,3))
        with self.assertRaises(ValueError):
            sample_video_for_vae(torch.zeros(5,1,1,3), 0)


class PoolingTests(unittest.TestCase):
    def test_cpu_pool_matches_existing_fp32_arithmetic(self):
        torch.manual_seed(19)
        z = torch.randn(1,24,3,20,28).half()
        expected = F.adaptive_avg_pool3d(z.float(), (2,8,8)).to(z.dtype)
        actual = pool_latent(z,2,8,8)
        self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
        self.assertIs(pool_latent(z,3,20,28), z)

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires MPS")
    def test_mps_pool_and_visual_audio_blur_match_cpu_bytes(self):
        torch.manual_seed(20)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for shape in ((1,24,3,20,28), (1,32,2,19)):
                with self.subTest(dtype=dtype, shape=shape):
                    z = torch.randn(shape).to(dtype)
                    gpu = z.to('mps')
                    functions = [_blur_latent]
                    if len(shape) == 5:
                        functions.append(lambda x: pool_latent(x,2,8,8))
                    for function in functions:
                        expected = function(z)
                        actual = function(gpu)
                        self.assertEqual(actual.device, gpu.device)
                        self.assertEqual(actual.dtype, dtype)
                        self.assertTrue(torch.equal(actual.cpu().view(torch.uint8),
                                                    expected.view(torch.uint8)))
        one = torch.ones(1,32,2,1,device='mps')
        self.assertEqual(_blur_latent(one).device, one.device)

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires MPS")
    def test_explicit_cpu_transfers_preserve_autograd(self):
        for function in (_blur_latent, lambda x: pool_latent(x,2,8,8)):
            cpu = torch.randn(1,2,3,20,28,requires_grad=True)
            gpu = cpu.detach().to('mps').requires_grad_()
            function(cpu).square().sum().backward()
            function(gpu).square().sum().backward()
            torch.testing.assert_close(gpu.grad.cpu(), cpu.grad, rtol=1e-5, atol=1e-6)


class TimingTests(unittest.TestCase):
    def test_synchronizes_only_the_selected_backend(self):
        with patch.object(torch.mps, 'synchronize') as mps, \
                patch.object(torch.cuda, 'synchronize') as cuda:
            synchronize('cpu')
            mps.assert_not_called()
            cuda.assert_not_called()
            synchronize('mps')
            mps.assert_called_once_with()
            synchronize('cuda:1')
            cuda.assert_called_once_with(torch.device('cuda:1'))


if __name__ == '__main__':
    unittest.main()
