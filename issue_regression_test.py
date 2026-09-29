"""Issue #21-23 regressions. Run with ComfyUI's Python; no model weights needed.

    python issue_regression_test.py
"""
import asyncio
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from comfy.cli_args import args
args.cpu = True
import folder_paths
import torch
with tempfile.TemporaryDirectory() as import_dir:
    with patch.object(folder_paths, "models_dir", import_dir):
        N = importlib.import_module(f"custom_nodes.{ROOT.name}.nodes")
import execution
import nodes as comfy_nodes
COMMON = importlib.import_module(N.__package__ + ".common")


class IssueRegressions(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "refmods"
        self.root.mkdir()
        N._MOD_CACHE.clear()
        N._MOD_CACHE_STAMPS.clear()
        self.addCleanup(N._MOD_CACHE.clear)
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_creators_are_queue_outputs_without_save_node(self):
        for cls in (N.MiniMaxH3RefModExtract, N.MiniMaxH3RefModMasterExtract):
            cls.INPUT_TYPES()  # initialize the V3 compatibility attributes
            schema = cls.define_schema()
            self.assertTrue(schema.is_output_node)
            self.assertEqual(len(schema.outputs), 2)
            inputs = {}
            for key, (kind, options) in cls.INPUT_TYPES()["required"].items():
                if "default" in options:
                    inputs[key] = options["default"]
            prompt = {"create": {"class_type": schema.node_id, "inputs": inputs}}
            with patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS, {schema.node_id: cls}):
                result = asyncio.run(execution.validate_prompt("creator", prompt, None))
            self.assertTrue(result[0], result)
            self.assertIn("create", result[2])

    def test_create_details_report_save_and_no_save(self):
        vae = types.SimpleNamespace(encode=lambda pixels: torch.ones(1,24,1,4,4))
        for save in (False, True):
            with patch.object(COMMON, "refmods_dir", return_value=str(self.root)):
                result = N.MiniMaxH3RefModExtract.execute(
                    "details", mode="encode", refs_image={"ref_image_0": torch.zeros(1,64,64,3)},
                    vae=vae, save=save)
            details = json.loads(result[1])
            self.assertEqual(len(details["saved_paths"]), int(save))
            self.assertEqual(details["tokens"], result[0][0][0].token_count)
            self.assertEqual(result.ui.as_dict()["text"], (result[1],))

    def test_mask_list_keeps_sizes_and_maps_each_reference(self):
        masks = [torch.zeros(1,64,96), torch.ones(1,96,64)]
        collected = N.MiniMaxH3RefModMaskList().collect(masks)[0]
        self.assertEqual([tuple(m.shape) for m in collected], [(1,64,96),(1,96,64)])
        refs = {"ref_image_0": torch.zeros(1,64,96,3), "ref_image_1": torch.zeros(1,96,64,3)}
        seen = []
        def mask_latent(z, mask_px, *args, **kwargs):
            seen.append(mask_px.clone())
            return z
        vae = types.SimpleNamespace(encode=lambda pixels: torch.ones(1,24,1,4,4))
        for cls in (N.MiniMaxH3RefModExtract, N.MiniMaxH3RefModMasterExtract):
            seen.clear()
            with patch.object(N, "_mask_latent", side_effect=mask_latent):
                cls.execute("masked", mode="training", identity=0, refs_image=refs,
                            vae=vae, mask_list=collected, save=False)
            self.assertEqual(len(seen), 2)
            self.assertEqual([tuple(m.shape) for m in seen], [(1,320,480),(1,480,320)])
            self.assertEqual([m.mean().item() for m in seen], [0,1])

    def test_mask_list_consumed_once_by_comfy_execution(self):
        masks = [torch.zeros(1,64,96), torch.ones(1,96,64)]
        node = N.MiniMaxH3RefModMaskList()
        result = asyncio.run(execution._async_map_node_over_list(
            "masks", "collect", node, {"masks": masks}, node.FUNCTION))
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0][0]), 2)

    def test_mixed_aspect_masks_follow_reference_resize_and_crop(self):
        masks = [torch.linspace(0,1,640).expand(320,640).unsqueeze(0),
                 torch.linspace(0,1,640).view(640,1).expand(640,320).unsqueeze(0)]
        refs = {f"ref_image_{i}": mask.unsqueeze(-1).expand(-1,-1,-1,3)
                for i, mask in enumerate(masks)}
        pixels_seen, masks_seen = [], []
        def encode(pixels):
            pixels_seen.append(pixels[...,0].clone())
            return torch.ones(1,24,1,pixels.shape[1]//16,pixels.shape[2]//16)
        def apply_mask(z, mask_px, *args, **kwargs):
            masks_seen.append(mask_px.clone())
            return z
        for mode in ("training", "encode"):
            pixels_seen.clear()
            masks_seen.clear()
            with patch.object(N, "_mask_latent", side_effect=apply_mask):
                N.MiniMaxH3RefModExtract.execute(
                    "alignment", mode=mode, identity=0, refs_image=refs,
                    mask_list=masks, vae=types.SimpleNamespace(encode=encode), save=False)
            for pixels, mask in zip(pixels_seen, masks_seen):
                # Lanczos/PIL images and float bilinear masks differ by up to
                # two uint8 levels from quantization and interpolation kernels.
                torch.testing.assert_close(mask, pixels, atol=2/255, rtol=0)

    def test_mask_broadcast_batch_and_mapping_errors(self):
        vae = types.SimpleNamespace(encode=lambda pixels: torch.ones(1,24,1,4,4))
        args = dict(name="masks", mode="encode", vae=vae, save=False,
                    refs_image={"ref_image_0": torch.zeros(1,64,64,3),
                                "ref_image_1": torch.zeros(1,64,64,3)})
        for kwargs in ({"mask": torch.ones(1,64,64)},
                       {"mask": torch.ones(2,64,64)},
                       {"mask_list": [torch.ones(1,64,64)]}):
            self.assertEqual(len(N.MiniMaxH3RefModExtract.execute(**args, **kwargs)[0]), 1)
        with patch.object(vae, "encode", side_effect=AssertionError("must validate before encoding")):
            with self.assertRaisesRegex(ValueError, "3 entries"):
                N.MiniMaxH3RefModExtract.execute(**args, mask_list=[torch.ones(3,64,64)])
            with self.assertRaisesRegex(ValueError, "not both"):
                N.MiniMaxH3RefModExtract.execute(**args, mask=torch.ones(1,64,64),
                                               mask_list=[torch.ones(1,64,64)])

    def test_encode_samples_causal_count_without_losing_clip_endpoint(self):
        clip = torch.arange(101).reshape(101,1,1,1).expand(-1,64,64,3).float() / 100
        seen = []
        def encode(pixels):
            seen.append(pixels[:,0,0,0].clone())
            n = pixels.shape[0]
            return torch.ones(1,24,1 if n == 1 else (n-5)//17*5+2,4,4)
        for limit, expected in ((16,5),(17,5),(22,22),(39,39),(1,1),(4,1),(101,90)):
            N.MiniMaxH3RefModExtract.execute(
                "motion", mode="encode", refs_video={"ref_video_0":clip},
                vae=types.SimpleNamespace(encode=encode), latent_frames=limit, save=False)
            self.assertEqual(len(seen[-1]), expected)
            self.assertEqual(seen[-1][0].item(), 0)
            if expected > 1:
                self.assertEqual(seen[-1][-1].item(), 1)

    def test_valid_22_frame_reference_reaches_vae_without_trimming(self):
        clip = torch.arange(22).reshape(22,1,1,1).expand(-1,32,32,3).float() / 21
        seen = []
        vae = types.SimpleNamespace(encode=lambda pixels:
            seen.append(pixels.clone()) or torch.ones(1,24,7,4,4))
        # Isolate temporal policy from image-resize quantization.
        with patch.object(N, '_resize_ref', side_effect=lambda src,*a:src), \
                patch.object(N, '_ensure_min_size', side_effect=lambda src:src):
            for mode in ('encode', 'training'):
                N.MiniMaxH3RefModExtract.execute(
                    'aligned', mode=mode, refs_video={'ref_video_0':clip},
                    vae=vae, latent_frames=22, identity=0, save=False)
                self.assertTrue(torch.equal(seen[-1], clip))

    def test_motion_only_realigns_differences_and_keeps_last_motion(self):
        clip = torch.linspace(0,1,23).square().reshape(23,1,1,1).expand(-1,32,32,3)
        seen = []
        vae = types.SimpleNamespace(encode=lambda pixels:
            seen.append(pixels.clone()) or torch.ones(1,24,7,4,4))
        with patch.object(N, '_resize_ref', side_effect=lambda src,*a:src), \
                patch.object(N, '_ensure_min_size', side_effect=lambda src:src):
            N.MiniMaxH3RefModExtract.execute(
                'motion', mode='training', refs_video={'ref_video_0':clip},
                vae=vae, latent_frames=2, identity=0, motion_only=True, save=False)
        expected = (clip[1:]-clip[:-1]).abs()
        expected /= expected.max()
        self.assertTrue(torch.equal(seen[-1], expected))

    def test_cli_samples_the_same_frames_as_the_node_helper(self):
        sys.path.insert(0, str(ROOT))
        self.addCleanup(sys.path.remove, str(ROOT))
        cli = importlib.import_module('extract_mod')
        import comfy.sd
        import comfy.utils
        clip = torch.arange(101).reshape(101,1,1,1).expand(-1,32,32,3).float()/100
        seen = []
        vae = types.SimpleNamespace(throw_exception_if_invalid=lambda:None,
            encode=lambda pixels:seen.append(pixels.clone()) or torch.ones(1,24,7,4,4))
        with patch.object(cli, '_load_video', return_value=clip), \
                patch.object(cli, '_resize_ref', side_effect=lambda src,*a:src), \
                patch.object(cli, 'ensure_min_size', side_effect=lambda src:src), \
                patch.object(comfy.sd, 'VAE', return_value=vae), \
                patch.object(comfy.utils, 'load_torch_file', return_value=({},{})):
            for mode, limit in (('encode',16),('encode',22),('training',2)):
                argv = ['extract_mod.py','--video','test.mp4','--vae','stub.safetensors',
                        '--device','cpu','--mode',mode,'--latent-frames',str(limit),
                        '--identity','0','--output',str(self.root),'--name','cli']
                with patch.object(sys, 'argv', argv):
                    cli.main()
                expected = COMMON.sample_video_for_vae(clip,limit if mode=='encode' else None)
                self.assertTrue(torch.equal(seen[-1], expected))


if __name__ == "__main__":
    unittest.main()
