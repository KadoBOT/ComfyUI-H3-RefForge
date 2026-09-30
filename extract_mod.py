"""
extract_mod.py — standalone MiniMax H3 RefMod extractor (no training)

Turns reference images/videos into a ``.safetensors`` mod that the ComfyUI
Load H3 RefMods / Apply nodes can inject into MiniMax H3 generation.  Only
the H3 video VAE is needed; the 29B DiT is never loaded.  Extraction runs the
Create H3 RefMod node itself, so a mod made here matches one made in a graph
with the same settings.

Two modes:

  * ``training`` (default) — refs are resized to a small latent grid
    (``--pool``, 16 px per cell, aspect-fit) and VAE-encoded at that size: a
    faithful low-resolution copy at a fraction of the tokens.  (Old name:
    ``pooled``.)
  * ``encode`` — refs are resized to ``--resolution`` short edge (down only)
    and encoded at that resolution, exactly like the official ref2video node.
    This is what carries fine identity (a face, an outfit); files are
    ~0.2-1 MB per frame.  (Old name: ``full``.)

Usage
-----
  # full-res identity mod (recommended for characters)
  python custom_nodes/ComfyUI-H3-RefForge/extract_mod.py \
      --image char.png --vae path/to/h3_video_vae.safetensors \
      --name my_character --mode encode --resolution 1024

  # tiny concept/motion mod
  python custom_nodes/ComfyUI-H3-RefForge/extract_mod.py \
      --video dance.mp4 --vae path/to/h3_video_vae.safetensors \
      --name dance --mode training --pool 8 --latent-frames 7

Multi-reference concept (each ref becomes its own latent frame):

  python custom_nodes/ComfyUI-H3-RefForge/extract_mod.py \
      --image face_a.png --image face_b.png --image full.png \
      --video dance.mp4 --vae path/to/h3_video_vae.safetensors \
      --name disney_char --mode encode --resolution 1024

Output goes to ``ComfyUI/models/refmods/`` by default (created on first run,
next to loras/ and unet/).

Run it with the same Python/venv that runs ComfyUI (it imports ``comfy``
from the install the script lives in).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

import comfy.model_management
import comfy.sd
import comfy.utils

PACKAGE = f"custom_nodes.{os.path.basename(HERE)}"
common = importlib.import_module(PACKAGE + ".common")
core = importlib.import_module(PACKAGE + ".core")
nodes = importlib.import_module(PACKAGE + ".nodes")


def main():
    ap = argparse.ArgumentParser(
        description="Extract a MiniMax H3 RefMod from one or more images/videos "
                    "of the same concept (they are stacked into a video-kind mod)")
    ap.add_argument("--image", action="append", default=[], metavar="PATH",
                    help="reference image path (repeatable for multi-view concepts)")
    ap.add_argument("--video", action="append", default=[], metavar="PATH",
                    help="reference video path (repeatable)")
    ap.add_argument("--vae", required=True, help="MiniMax H3 video VAE .safetensors")
    ap.add_argument("--name", default=None, help="mod name (default: first source file stem)")
    ap.add_argument("--subfolder", default="", help="optional subfolder inside models/refmods")
    ap.add_argument("--output", default=None,
                    help="output dir (default: ComfyUI/models/refmods)")
    ap.add_argument("--mode", choices=["training", "encode", "full", "pooled"],
                    default="training",
                    help="training = refs encoded on a small latent grid (default); encode = full-res VAE encode (~1K tokens/img); full/pooled = old names, still accepted")
    ap.add_argument("--resolution", type=int, default=1024,
                    help="target short edge in px, downscale only (default 1024; 2048 = max fidelity). training mode never uses more than resolution/16 latent cells on the short edge")
    ap.add_argument("--pool", type=int, default=16, help="training mode: latent grid, 16 px per cell (even, default 16)")
    ap.add_argument("--pool-w", type=int, default=None, help="grid width (default: == --pool)")
    ap.add_argument("--latent-frames", type=int, default=16,
                    help="per-video limit (default 16; images use 1): training mode keeps at most this many latent frames (16 gives 12); encode mode samples this many source frames on H3's 17k+5 grid (16 selects 5, below 5 selects the first image)")
    ap.add_argument("--multiplier", type=int, default=1,
                    help="data multiplier: repeat the extracted ref N times along time so a short video/GIF isn't drowned out by the main video's tokens (default 1 = no repeat)")
    ap.add_argument("--max-tokens", type=int, default=5120,
                    help="hard cap on the total injected tokens (0 = off; default 5120): an image-only stack shrinks its grid, videos drop near-duplicate latent frames first, then resample the rest to fit")
    ap.add_argument("--max-edge", type=int, default=1536,
                    help="resize source so the longest edge is <= this while loading (default 1536)")
    ap.add_argument("--max-frames", type=int, default=60,
                    help="max video frames loaded (uniform) before sampling (default 60; lower for CPU)")
    ap.add_argument("--description", default="",
                    help="optional text describing the concept (e.g. 'a ginger woman with messy hair', "
                         "'an animation style'). Stored in the mod and emitted by the loaders so it "
                         "can be merged into the prompt without retyping it.")
    ap.add_argument("--concept-type", choices=list(core.CONCEPT_TYPES), default="generic",
                    help="what this mod represents (default: generic). 'identity' with a training grid "
                         "under 16 prints a warning: use --mode encode for people.")
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu (VAE device)")
    args = ap.parse_args()
    if not args.image and not args.video:
        ap.error("provide at least one --image or --video")

    device = comfy.model_management.get_torch_device() if args.device == "auto" \
        else torch.device(args.device)
    print(f"[extract] loading VAE {args.vae} (device={args.device})")
    sd, metadata = comfy.utils.load_torch_file(args.vae, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata, device=device)
    vae.throw_exception_if_invalid()

    # keep the load-time longest-edge cap ahead of the --resolution target
    load_max_edge = max(args.resolution * 2, args.max_edge)
    images = {f"ref_image_{i + 1}": common.load_image_file(path, max_edge=load_max_edge)
              for i, path in enumerate(args.image)}
    videos = {f"ref_video_{i + 1}": common.load_video_file(path, max_frames=args.max_frames, max_edge=load_max_edge)
              for i, path in enumerate(args.video)}
    name = args.name or os.path.splitext(os.path.basename((args.image or args.video)[0]))[0]
    result = nodes.MiniMaxH3RefModExtract.execute(
        name, mode=args.mode, refs_image=images, refs_video=videos, vae=vae,
        ref_resolution=args.resolution, pool_h=args.pool, pool_w=args.pool_w or args.pool,
        latent_frames=args.latent_frames, multiplier=args.multiplier,
        max_tokens=args.max_tokens, description=args.description,
        concept_type=args.concept_type, save=False)
    mod = result[0][0][0]
    # so Fix H3 RefMod Config can re-save in place
    mod.path = os.path.join(args.output, mod.name) if args.output else common.mod_output_path(mod.name, args.subfolder)
    path = mod.save(mod.path)
    mb = mod.latent.numel() * mod.latent.element_size() / 1024 / 1024
    print(f"[extract] saved {mod.kind} mod '{mod.name}' "
          f"({mod.token_count} tokens, {mb:.2f} MB) -> {path}")
    print(f"[extract] load it in ComfyUI with the Load H3 RefMods node "
          f"(dropdown '{mod.name}', strength 1.0 = full ref), then chain "
          f"Apply H3 RefMod (Cond) into your sampling.")


if __name__ == "__main__":
    # ComfyUI executes nodes under a global inference mode; mirror it.
    with torch.inference_mode():
        main()
