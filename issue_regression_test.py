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
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
COMMON = importlib.import_module(N.__package__ + ".common")
CORE = importlib.import_module(N.__package__ + ".core")
PROMPT = importlib.import_module(N.__package__ + ".prompt")


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
            # both refs are encoded on the first source's 4x6 grid
            self.assertEqual([tuple(m.shape) for m in seen], [(1,64,96),(1,64,96)])
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
        with patch.object(N, '_resize_ref', side_effect=lambda src,*a:src):
            for mode in ('encode', 'training'):
                N.MiniMaxH3RefModExtract.execute(
                    'aligned', mode=mode, refs_video={'ref_video_0':clip},
                    vae=vae, latent_frames=22, identity=0, save=False)
                self.assertTrue(torch.equal(seen[-1], clip))

    def test_motion_only_realigns_differences_and_keeps_last_motion(self):
        clip = torch.linspace(0,1,23).square().reshape(23,1,1,1).expand(-1,32,32,3)
        seen = []
        vae = types.SimpleNamespace(encode=lambda pixels:
            seen.append(pixels.clone()) or torch.ones(1,24,2,2,2))
        N.MiniMaxH3RefModExtract.execute(
            'motion', mode='training', refs_video={'ref_video_0':clip},
            vae=vae, latent_frames=2, identity=0, motion_only=True, save=False)
        expected = (clip[1:]-clip[:-1]).abs()
        expected /= expected.max()
        # two latent frames come from five source frames
        self.assertTrue(torch.equal(seen[-1], COMMON.sample_video_for_vae(expected, 5)))

    def test_cli_samples_the_same_frames_as_the_node_helper(self):
        sys.path.insert(0, str(ROOT))
        self.addCleanup(sys.path.remove, str(ROOT))
        cli = importlib.import_module('extract_mod')
        import comfy.sd
        import comfy.utils
        clip = torch.arange(101).reshape(101,1,1,1).expand(-1,32,32,3).float()/100
        seen = []
        vae = types.SimpleNamespace(throw_exception_if_invalid=lambda:None,
            encode=lambda pixels:seen.append(pixels.clone()) or torch.ones(1,24,7,2,2))
        with patch.object(COMMON, 'load_video_file', return_value=clip), \
                patch.object(N, '_resize_ref', side_effect=lambda src,*a:src), \
                patch.object(comfy.sd, 'VAE', return_value=vae), \
                patch.object(comfy.utils, 'load_torch_file', return_value=({},{})):
            for mode, limit, frames in (('encode',16,16),('encode',22,22),('training',2,5),('training',7,22)):
                argv = ['extract_mod.py','--video','test.mp4','--vae','stub.safetensors',
                        '--device','cpu','--mode',mode,'--latent-frames',str(limit),
                        '--output',str(self.root),'--name','cli']
                with patch.object(sys, 'argv', argv):
                    cli.main()
                self.assertTrue(torch.equal(seen[-1], COMMON.sample_video_for_vae(clip, frames)))
                self.assertTrue((self.root / 'cli.safetensors').exists())

    def test_compressed_reference_encodes_the_resized_image_on_its_grid(self):
        z = torch.randn(1,24,1,10,16)
        seen = []
        vae = types.SimpleNamespace(encode=lambda pixels: seen.append(pixels.shape) or z)
        image = torch.rand(1,512,768,3)
        mod = N.MiniMaxH3RefModExtract.execute(
            'grid', mode='training', refs_image={'ref_image_1': image}, vae=vae, save=False)[0][0][0]
        # a 16 grid fits a 2:3 source as 10x16 cells of 16 px; the encode is stored as is
        self.assertEqual(seen, [(1,160,256,3)])
        self.assertTrue(torch.equal(mod.latent, z.half()))
        self.assertEqual((mod.latent_h, mod.latent_w, mod.optimize_steps), (10,16,0))

    def test_compressed_grid_never_upscales_a_small_source(self):
        seen = []
        vae = types.SimpleNamespace(encode=lambda pixels:
            seen.append(pixels.clone()) or torch.ones(1,24,1,pixels.shape[1]//16,pixels.shape[2]//16))
        image = torch.rand(1,64,96,3)
        mod = N.MiniMaxH3RefModExtract.execute(
            'small', mode='training', refs_image={'ref_image_1': image}, vae=vae,
            pool_h=32, pool_w=32, save=False)[0][0][0]
        self.assertTrue(torch.equal(seen[0], image))
        self.assertEqual((mod.latent_h, mod.latent_w), (4,6))

    def test_image_stack_budget_shrinks_the_grid_instead_of_dropping_refs(self):
        seen = []
        vae = types.SimpleNamespace(encode=lambda pixels:
            seen.append(pixels.shape) or torch.ones(1,24,1,pixels.shape[1]//16,pixels.shape[2]//16))
        refs = {f'ref_image_{i}': torch.rand(1,256,256,3) for i in range(1, 5)}
        for mode in ('training', 'encode'):
            seen.clear()
            mod = N.MiniMaxH3RefModExtract.execute(
                'budget', mode=mode, refs_image=refs, vae=vae, max_tokens=100, save=False)[0][0][0]
            self.assertEqual(mod.latent_t, 4)
            self.assertEqual(mod.token_count, 100)
            self.assertEqual(seen, [(1,160,160,3)] * 4)

    def test_merge_averages_every_reference_encode(self):
        values = iter((1.0, 3.0))
        vae = types.SimpleNamespace(encode=lambda pixels: torch.full((1,24,1,4,4), next(values)))
        refs = {'ref_image_1': torch.rand(1,64,64,3), 'ref_image_2': torch.rand(1,64,64,3)}
        mod = N.MiniMaxH3RefModExtract.execute(
            'merged', mode='training', refs_image=refs, vae=vae, merge=True, save=False)[0][0][0]
        self.assertTrue(torch.equal(mod.latent, torch.full((1,24,1,4,4), 2.0).half()))

    def test_decode_visual_decodes_a_lone_latent_as_a_two_frame_clip(self):
        seen = []
        def decode(latent):
            seen.append(latent.shape[2])
            frames = 1 if latent.shape[2] == 1 else (latent.shape[2] - 2) // 5 * 17 + 5
            return torch.arange(frames).float().reshape(1,frames,1,1,1).expand(-1,-1,32,32,3)
        vae = types.SimpleNamespace(decode=decode)
        image = COMMON.decode_visual(vae, torch.zeros(1,24,1,2,2))
        video = COMMON.decode_visual(vae, torch.zeros(1,24,7,2,2))
        self.assertEqual(seen, [2, 7])
        self.assertEqual((tuple(image.shape), image.max().item()), ((1,32,32,3), 0))
        self.assertEqual(tuple(video.shape), (22,32,32,3))

    def test_text_encode_caps_the_vision_tokens_of_a_stack_view(self):
        def decode(latent):
            t, h, w = latent.shape[2:]
            return torch.rand(1, 4 * t - 3, 16 * h, 16 * w, 3)
        vae = types.SimpleNamespace(first_stage_model=object.__new__(MiniMaxH3VideoVAE), decode=decode)
        def view(source, h, w):
            mod = CORE.H3RefMod(name=source, kind='video', latent=torch.zeros(1,24,6,h,w),
                                latent_h=h, latent_w=w, latent_t=6, source=source)
            items, blocks = PROMPT.prepare_references([(mod, 1.0)], vae)
            self.assertTrue(torch.equal(blocks[0]['latent'], mod.latent))
            self.assertEqual(items[0]['timestamps'], [0.0, 0.5])
            return tuple(items[0]['data'].shape)
        # grid 36 decodes to 384x576 px, 216 Qwen tokens per frame; a stack's view gets 54
        self.assertEqual(view('stack', 36, 24), (2,288,192,3))
        self.assertEqual(view('video', 36, 24), (2,576,384,3))
        self.assertEqual(view('stack', 16, 10), (2,256,160,3))

    def test_reference_map_carries_each_description_on_one_line(self):
        def mod(name, kind, description):
            latent = torch.zeros(1,32,2,4) if kind == 'audio' else torch.zeros(1,24,1,2,2)
            return CORE.H3RefMod(name=name, kind=kind, latent=latent, description=description)
        mods = [(mod('alice', 'image', 'adult woman,\n  long brown hair'), 1.0), (mod('room', 'image', ' '), 0.5),
                (mod('voice', 'audio', 'warm low voice'), 1.0), (mod('off', 'image', 'unused'), 0.0)]
        self.assertEqual(PROMPT.reference_map(mods), "<Picture 1> = alice: adult woman, long brown hair\n"
                         "<Picture 2> = room\n<Audio 1> = voice: warm low voice")


if __name__ == "__main__":
    unittest.main()
