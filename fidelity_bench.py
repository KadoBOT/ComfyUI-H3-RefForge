"""Measure how faithfully Compressed Reference RefMods keep their source, per token.

Runs the RefMod 0.2.6 algorithm (encode at ref_resolution, average-pool the
latent to the grid, 500 Adam steps on the trilinear latent MSE; copied below
as the baseline) and the current Create H3 RefMod node on the same sources.
Every latent is decoded with the real H3 video VAE and scored against the part
of the source it stores (0.2.6 stretched the source onto the grid, the node
cover-crops it):

  images  PSNR/SSIM at the reference resolution (``hi``: decode upsampled
          bicubically) and at the latent's own pixel size (``lo``)
  videos  mean ``lo`` PSNR/SSIM over the decoded frames, each against the
          source frame at the same relative time in the clip
  stack   the first four images as one stacked mod under a max_tokens
          budget: references kept and their mean ``hi`` score

The DiT reads the stored latent directly, so decoded scores are a proxy for
what it can recover, not a generation benchmark.

    python custom_nodes/ComfyUI-H3-RefForge/fidelity_bench.py \
        --vae models/vae/minimax-h3/minimax_h3_video_vae_int8_convrot.safetensors \
        --image a.png --image b.jpg --video c.mp4 --out bench.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

import comfy.sd
import comfy.utils

PACKAGE = f"custom_nodes.{os.path.basename(HERE)}"
common = importlib.import_module(PACKAGE + ".common")
core = importlib.import_module(PACKAGE + ".core")
nodes = importlib.import_module(PACKAGE + ".nodes")


# ── RefMod 0.2.6 Compressed Reference (the baseline) ─────────────────────

def legacy_ensure_min_size(image, floor=320):
    h, w = image.shape[1], image.shape[2]
    if h >= floor and w >= floor:
        return image
    scale = floor / min(h, w)
    tw = max(floor, round(w * scale / 32) * 32)
    th = max(floor, round(h * scale / 32) * 32)
    samples = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), tw, th, "lanczos", "disabled")
    return samples.movedim(1, -1)


def legacy_optimize_latent(z_small, z_full, steps, lr=0.02):
    with torch.inference_mode(False), torch.set_grad_enabled(True):
        target = z_full.clone().float()
        param = torch.nn.Parameter(z_small.clone().float())
        opt = torch.optim.Adam([param], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            up = F.interpolate(param, size=tuple(target.shape[2:]), mode="trilinear", align_corners=False)
            F.mse_loss(up, target).backward()
            opt.step()
        return param.detach().to(z_small.dtype)


def legacy_encode(vae, src, is_video, resolution):
    """The full-resolution encode 0.2.6 pooled from; every grid of a source shares it."""
    src = legacy_ensure_min_size(common.resize_ref(src, resolution))
    if is_video:
        src = common.sample_video_for_vae(src)
    return vae.encode(src)


def legacy_compress(z, grid, latent_t, aspect, steps):
    gh, gw = core.aspect_grid(grid, grid, aspect)
    small = core.pool_latent(z, min(latent_t, z.shape[2]), gh, gw).to(torch.float16)
    return legacy_optimize_latent(small, z.float(), steps)


# ── current node ─────────────────────────────────────────────────────────

def extract(vae, grid, resolution, latent_frames=16, max_tokens=0, images=(), videos=()):
    with contextlib.redirect_stdout(io.StringIO()):
        return nodes.MiniMaxH3RefModExtract.execute(
            "bench", mode="training", vae=vae, ref_resolution=resolution, pool_h=grid, pool_w=grid,
            latent_frames=latent_frames, max_tokens=max_tokens, save=False,
            refs_image={f"ref_image_{i + 1}": x for i, x in enumerate(images)},
            refs_video={f"ref_video_{i + 1}": x for i, x in enumerate(videos)})[0][0][0].latent


# ── scoring ──────────────────────────────────────────────────────────────

def decode(vae, latent):
    return common.decode_visual(vae, latent).float().cpu().clamp(0.0, 1.0)


def scores(pred, target):
    """Mean PSNR and SSIM of two [T, H, W, 3] clips in [0, 1]."""
    psnr, ssim = [], []
    for a, b in zip(pred.numpy(), target.numpy()):
        psnr.append(10.0 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12)))
        ssim.append(structural_similarity(a, b, channel_axis=2, data_range=1.0))
    return statistics.fmean(psnr), float(statistics.fmean(ssim))


def resize(x, w, h):
    return comfy.utils.common_upscale(x[..., :3].movedim(-1, 1), w, h, "lanczos", "disabled").movedim(1, -1).clamp(0.0, 1.0)


def upsample(x, h, w):
    return F.interpolate(x.movedim(-1, 1), size=(h, w), mode="bicubic", align_corners=False).movedim(1, -1).clamp(0.0, 1.0)


def center_crop(x, h, w):
    """The part of ``x`` a cover-crop onto an ``h x w`` canvas keeps (as comfy.utils.common_upscale)."""
    ih, iw = x.shape[1], x.shape[2]
    old, new = iw / ih, w / h
    if old > new:
        dx = round((iw - iw * (new / old)) / 2)
        return x[:, :, dx:iw - dx]
    if old < new:
        dy = round((ih - ih * (old / new)) / 2)
        return x[:, dy:ih - dy]
    return x


def score_image(vae, z, x_ref, crop):
    gh, gw = z.shape[3], z.shape[4]
    region = center_crop(x_ref, gh, gw) if crop else x_ref
    pixels = decode(vae, z)
    hi = scores(upsample(pixels, region.shape[1], region.shape[2]), region)
    lo = scores(pixels, resize(region, gw * 16, gh * 16))
    return hi, lo


def score_video(vae, z, clip, crop):
    gh, gw = z.shape[3], z.shape[4]
    pixels = decode(vae, z)
    n = pixels.shape[0]
    frames = clip[[round(i * (clip.shape[0] - 1) / max(1, n - 1)) for i in range(n)]]
    region = center_crop(frames, gh, gw) if crop else frames
    return scores(pixels, resize(region, gw * 16, gh * 16))


def tokens(z):
    return z.shape[2] * (z.shape[3] // 2) * (z.shape[4] // 2)


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vae", required=True)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument("--grids", type=int, nargs="+", default=[8, 12, 16, 24, 32, 48], help="image grid long edges (latent cells)")
    ap.add_argument("--video-grids", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--latent-frames", type=int, nargs="+", default=[2, 7, 16], help="video latent_frames settings")
    ap.add_argument("--resolution", type=int, default=1024, help="reference short edge, as ref_resolution")
    ap.add_argument("--steps", type=int, default=500, help="0.2.6 refinement steps (its default)")
    ap.add_argument("--stack-grid", type=int, default=16)
    ap.add_argument("--stack-budget", type=int, default=128, help="max_tokens for the stack case (0 skips it)")
    ap.add_argument("--out", default=None, help="write every result as JSON")
    args = ap.parse_args()
    if not args.image and not args.video:
        ap.error("provide at least one --image or --video")

    sd, metadata = comfy.utils.load_torch_file(args.vae, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    rows = []

    refs = [(os.path.basename(p), common.resize_ref(common.load_image_file(p), args.resolution)) for p in args.image]
    for name, x_ref in refs:
        z_full, enc_s = timed(legacy_encode, vae, x_ref, False, args.resolution)
        single = scores(vae.decode(z_full).reshape(-1, *x_ref.shape[1:])[:1].float().cpu().clamp(0.0, 1.0), x_ref)
        full = scores(decode(vae, z_full), x_ref)
        rows.append({"case": "image", "source": name, "method": "full_encode", "tokens": tokens(z_full),
                     "hi_psnr": full[0], "hi_ssim": full[1], "single_decode_psnr": single[0], "single_decode_ssim": single[1]})
        print(f"{name[:30]:30s} full encode {tokens(z_full):5d} tok  {full[0]:6.2f} dB {full[1]:.4f} "
              f"(single-latent decode {single[0]:6.2f} dB {single[1]:.4f})", flush=True)
        for g in args.grids:
            old, old_s = timed(legacy_compress, z_full, g, 1, x_ref.shape[1] / x_ref.shape[2], args.steps)
            new, new_s = timed(extract, vae, g, args.resolution, images=[x_ref])
            for method, z, crop, seconds in (("0.2.6", old, False, enc_s + old_s), ("new", new, True, new_s)):
                hi, lo = score_image(vae, z, x_ref, crop)
                rows.append({"case": "image", "source": name, "method": method, "grid": g, "latent": list(z.shape[2:]),
                             "tokens": tokens(z), "hi_psnr": hi[0], "hi_ssim": hi[1], "lo_psnr": lo[0], "lo_ssim": lo[1],
                             "seconds": seconds})
                print(f"{name[:30]:30s} grid {g:2d} {method:5s} {tokens(z):5d} tok  hi {hi[0]:6.2f} dB {hi[1]:.4f}  "
                      f"lo {lo[0]:6.2f} dB {lo[1]:.4f}  {seconds:6.2f}s", flush=True)

    for path in args.video:
        name = os.path.basename(path)
        clip = common.resize_ref(common.load_video_file(path), args.resolution)
        z_full, enc_s = timed(legacy_encode, vae, clip, True, args.resolution)
        for g in args.video_grids:
            for lf in args.latent_frames:
                old, old_s = timed(legacy_compress, z_full, g, lf, clip.shape[1] / clip.shape[2], args.steps)
                new, new_s = timed(extract, vae, g, args.resolution, latent_frames=lf, videos=[clip])
                for method, z, crop, seconds in (("0.2.6", old, False, enc_s + old_s), ("new", new, True, new_s)):
                    lo = score_video(vae, z, clip, crop)
                    rows.append({"case": "video", "source": name, "frames": clip.shape[0], "method": method, "grid": g,
                                 "latent_frames": lf, "latent": list(z.shape[2:]), "tokens": tokens(z),
                                 "lo_psnr": lo[0], "lo_ssim": lo[1], "seconds": seconds})
                    print(f"{name[:30]:30s} grid {g:2d} lf {lf:2d} {method:5s} {tokens(z):5d} tok  "
                          f"lo {lo[0]:6.2f} dB {lo[1]:.4f}  {seconds:6.2f}s", flush=True)

    if args.stack_budget and len(refs) >= 4:
        stack = [x for _, x in refs[:4]]
        aspect = stack[0].shape[1] / stack[0].shape[2]
        frames = [legacy_compress(legacy_encode(vae, x, False, args.resolution), args.stack_grid, 1, aspect, args.steps) for x in stack]
        with contextlib.redirect_stdout(io.StringIO()):
            kept = core.fit_token_budget(torch.cat(frames, dim=2), args.stack_budget, "bench")
        old_kept = [i for i, z in enumerate(frames) if any(torch.equal(z[:, :, 0], kept[:, :, t]) for t in range(kept.shape[2]))]
        new = extract(vae, args.stack_grid, args.resolution, max_tokens=args.stack_budget, images=stack)
        for method, latent, members, crop in (("0.2.6", kept, old_kept, False), ("new", new, range(4), True)):
            hi = [score_image(vae, latent[:, :, t:t + 1], stack[i], crop)[0] for t, i in enumerate(members)]
            row = {"case": "stack", "method": method, "grid": args.stack_grid, "budget": args.stack_budget,
                   "latent": list(latent.shape[2:]), "tokens": tokens(latent), "refs_kept": len(hi),
                   "hi_psnr": statistics.fmean(p for p, _ in hi), "hi_ssim": statistics.fmean(s for _, s in hi)}
            rows.append(row)
            print(f"stack of 4, max_tokens {args.stack_budget}: {method:5s} keeps {row['refs_kept']}/4 refs "
                  f"{row['latent']} {row['tokens']} tok, kept refs hi {row['hi_psnr']:6.2f} dB {row['hi_ssim']:.4f}", flush=True)

    images = [r for r in rows if r["case"] == "image" and r["method"] != "full_encode"]
    if images:
        print("\nimages, mean over sources (hi = vs the source at reference resolution, lo = at the latent's pixel size)")
        print(f"{'grid':>4s} {'method':6s} {'tokens':>7s} {'hi PSNR':>8s} {'hi SSIM':>8s} {'lo PSNR':>8s} {'lo SSIM':>8s} {'sec':>6s}")
        for g in args.grids:
            for method in ("0.2.6", "new"):
                sel = [r for r in images if r["grid"] == g and r["method"] == method]
                mean = lambda key: statistics.fmean(r[key] for r in sel)
                print(f"{g:4d} {method:6s} {mean('tokens'):7.1f} {mean('hi_psnr'):8.2f} {mean('hi_ssim'):8.4f} "
                      f"{mean('lo_psnr'):8.2f} {mean('lo_ssim'):8.4f} {mean('seconds'):6.2f}")
        full = [r for r in rows if r["method"] == "full_encode"]
        print(f"full encode: {statistics.fmean(r['tokens'] for r in full):.0f} tokens, "
              f"{statistics.fmean(r['hi_psnr'] for r in full):.2f} dB {statistics.fmean(r['hi_ssim'] for r in full):.4f} "
              f"(single-latent decode {statistics.fmean(r['single_decode_psnr'] for r in full):.2f} dB)")
    videos = [r for r in rows if r["case"] == "video"]
    if videos:
        print("\nvideos, mean over clips (lo = decoded frames vs source frames at the latent's pixel size)")
        print(f"{'grid':>4s} {'lf':>3s} {'method':6s} {'tokens':>7s} {'lo PSNR':>8s} {'lo SSIM':>8s} {'sec':>7s}")
        for g in args.video_grids:
            for lf in args.latent_frames:
                for method in ("0.2.6", "new"):
                    sel = [r for r in videos if r["grid"] == g and r["latent_frames"] == lf and r["method"] == method]
                    mean = lambda key: statistics.fmean(r[key] for r in sel)
                    print(f"{g:4d} {lf:3d} {method:6s} {mean('tokens'):7.0f} {mean('lo_psnr'):8.2f} "
                          f"{mean('lo_ssim'):8.4f} {mean('seconds'):7.2f}")
    if args.out:
        env = {"torch": torch.__version__, "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
               "vae": os.path.basename(args.vae), "resolution": args.resolution, "steps": args.steps}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"env": env, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    # ComfyUI executes nodes under a global inference mode; mirror it.
    with torch.inference_mode():
        main()
