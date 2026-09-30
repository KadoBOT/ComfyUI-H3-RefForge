# RefMod creation audit

Scope: visual RefMod creation in ComfyUI-MiniMaxH3Mod at `f946208` (0.2.6 plus
unreleased fixes): Create/Master in `nodes.py`, `core.py`, `common.py`,
`prompt.py` and `extract_mod.py`. Goal: RefMods closer to their source at the
same or a lower reference-token cost.

## Results

`fidelity_bench.py` runs the 0.2.6 Compressed Reference algorithm (encode at
`ref_resolution`, average-pool the latent to the grid, 500 refinement steps)
and the current Create node on the same sources. Setup: H3 video VAE
(`minimax_h3_video_vae_int8_convrot`), `ref_resolution` 1024, RTX 5090 Laptop
GPU, torch 2.14. Sources: 10 images (portrait and press photos, character
sheets, a location render, pixel art, a 21:9 hero image) and 2 clips.
Tokens are latent frames × (H/2) × (W/2).

Each latent is decoded with the real VAE. `hi` scores the decode, upscaled
bicubically, against the source at reference resolution; `lo` scores it against
the source resized to the latent's pixel size (16 px per cell). The DiT reads
the latent directly, so these are proxies for what it can recover, not a
generation benchmark.

### Images, mean over 10

| Grid | Tokens | 0.2.6 hi PSNR / SSIM | New hi PSNR / SSIM | 0.2.6 lo PSNR / SSIM | New lo PSNR / SSIM |
| ---: | ---: | --- | --- | --- | --- |
| 8 | 11.6 | 14.53 / 0.4939 | 16.47 / 0.4808 | 15.27 / 0.2389 | 17.61 / 0.4452 |
| 12 | 25.2 | 14.61 / 0.4927 | 18.75 / 0.5244 | 15.15 / 0.2891 | 20.33 / 0.5820 |
| 16 | 42.4 | 14.87 / 0.4883 | 20.29 / 0.5555 | 15.33 / 0.3280 | 22.22 / 0.6715 |
| 24 | 100.8 | 16.13 / 0.4868 | 22.45 / 0.6353 | 16.51 / 0.4015 | 24.44 / 0.7637 |
| 32 | 182.4 | 17.10 / 0.4946 | 23.42 / 0.6764 | 17.41 / 0.4448 | 25.04 / 0.7766 |
| 48 | 400.8 | 17.97 / 0.4958 | 24.67 / 0.7350 | 18.08 / 0.4709 | 25.61 / 0.7891 |

Full Reference averages 1338 tokens and 26.89 dB / 0.8219.

Per image, the new path scores higher on hi PSNR, lo PSNR and lo SSIM in all
60 image/grid pairs. Hi SSIM is higher on 4/10 images at grid 8, 9/10 at 12,
8/10 at 16 and 10/10 at 24, 32 and 48.

Fewer tokens for the same fidelity: new grid 24 (101 tokens) beats 0.2.6 grid
48 (401 tokens) on both hi metrics for every image. New grid 16 (42 tokens)
beats it on hi PSNR for every image and on hi SSIM for 8 of 10.

Extraction takes 0.01–0.3 s instead of 0.5–2.2 s per image.

### Videos, mean over 2 clips (`lo`, each decoded frame against the source frame at the same relative time)

| Grid | `latent_frames` | 0.2.6 tokens | New tokens | 0.2.6 PSNR / SSIM | New PSNR / SSIM |
| ---: | ---: | ---: | ---: | --- | --- |
| 8 | 2 | 16 | 16 | 16.38 / 0.3065 | 18.51 / 0.4474 |
| 8 | 7 | 56 | 56 | 16.74 / 0.3143 | 19.41 / 0.4979 |
| 8 | 16 | 128 | 96 | 16.72 / 0.3135 | 19.73 / 0.5191 |
| 16 | 2 | 72 | 72 | 18.02 / 0.4463 | 22.35 / 0.6330 |
| 16 | 7 | 252 | 252 | 19.09 / 0.4602 | 24.30 / 0.7375 |
| 16 | 16 | 576 | 432 | 19.14 / 0.4649 | 24.84 / 0.7630 |

The new path is higher on both metrics for both clips at every setting.
Extraction takes 0.02–0.3 s instead of 33–44 s per clip.

### Stacked images under a budget

The first four images at grid 16 with `max_tokens=128` and `truncate`: 0.2.6
keeps 3 of the 4 references (3 × 16×10 cells, 120 tokens) and scores 14.72 dB
/ 0.4899 on those three. The new path keeps all four (4 × 14×8 cells, 112
tokens) at 18.96 dB / 0.4632.

## Fixed

1. **Pooled latents.** 0.2.6 average-pooled a full-resolution latent to the
   grid. An average of latent cells is not the latent of a smaller image, and
   decodes 2–7 dB further from the source than encoding the resized image.
   Compressed References are now resized in pixel space (area filter when
   shrinking) and encoded. `fit_grid` keeps the aspect ratio and never makes
   the grid finer than the source's 16 px cells or `ref_resolution`.
2. **Refinement.** `optimize_latent` (Refinement Steps, default 500) fitted a
   trilinear upsample of the small latent to the full latent, a latent-space
   objective whose optimum is the local average. Starting from the new grid
   encodes, it lowers hi PSNR on all 10 images: at grid 16 from 20.29 dB to
   15.45 dB after 50 steps and 14.66 dB after 500; at grid 32 from 23.42 dB to
   16.74 dB after 500. Removed together with `optimize_latent_multi`.
   `identity` stays in the schema because widget values are positional.
3. **Single-latent decode.** The H3 VAE reconstructs a one-frame latent much
   worse than the same frame inside a clip: 17.14 dB against 26.89 dB for the
   10 Full Reference encodes. Text Encode (the image Qwen sees) and Inspect used
   the one-frame path. `common.decode_visual` now decodes it as two repeated
   frames and keeps the first.
4. **Budgets dropped references.** Under `truncate`, `fit_token_budget` treats a
   stack's references as frames and removes whole ones. The Create node now
   plans the cost before encoding and shrinks the grid of an image-only stack
   so every reference fits. Videos still fit by dropping frames.
5. **Full video encodes.** 0.2.6 encoded the whole sampled clip at reference
   resolution, then pooled time and space. The clip is now sampled to the
   frames that give at most `latent_frames` causal latents (2, 7, 12, 17…; 16
   gives 39 frames and 12 latents), resized to the grid and encoded.
6. **Tiny references were enlarged.** `ensure_min_size` upscaled references
   under 320 px, so a 64 px image cost 100 tokens as a Full Reference and 64 as
   a Compressed one with no added detail. The VAE tiler handles small inputs
   (32×32 to 288×512 images and 5/22/39-frame clips encode and decode), so it
   is removed; that image now costs 4 tokens in both modes.
7. **Stretched stacks.** Compressed Reference stacks pooled every reference to
   the first one's grid regardless of aspect ratio. They are now cover-cropped
   like Full Reference stacks, and masks follow the same crop.
8. **Second pipeline in the CLI.** `extract_mod.py` re-implemented extraction
   without masks, merge, motion-only or presets. It now runs the Create node.
9. **Dead code and metadata.** Removed `gauntlet_harness.py`, the legacy
   `optimize` argument, the CLI's `--identity`, README references to test files
   that do not exist, the `/tests/` ignore rule that hid new tests, and a stale
   27 MB `node.zip` of 0.2.7. `pyproject.toml` (0.2.7) and `__init__.py`
   (0.2.6) now agree on 0.3.0.

## Tried and rejected

| Idea | Result |
| --- | --- |
| 0.2.6 refinement on top of the grid encode | Worse on every image (fix 2) |
| Refinement against decoded pixels | The int8 H3 decoder has no autograd path |
| A tiny autoencoder as a differentiable decoder | Local taeh3 weights do not match the latent layout (96 vs 256 channels) |
| Latent back-projection (re-encode the decode residual) | 0.1–1 dB worse |
| Pixel back-projection (adjust the thumbnail so its decode matches) | Best at iteration 0 |
| Lanczos or antialiased bicubic instead of area downscaling | Area 0.1 dB better, and area decodes score 0.002–0.019 higher face similarity to the source |
| Unsharp mask before encoding | Worse |
| Six picture blocks instead of one clip for a stack | −0.018 face identity in generated video, not significant; see [EVALUATION.md](EVALUATION.md) |
| Text Encode showing each stacked still as its own small vision block | +0.006 face identity, not significant, and 2.2 s slower per render |

## Open gaps

- **Generation benchmark covers one person.** The scores above measure the
  stored latent. [EVALUATION.md](EVALUATION.md) measures face identity in
  generated video for one six-photo stack; style, objects, single images, videos
  and voice remain unmeasured.
- **Grid 8 hi SSIM.** At the smallest grid, hi SSIM is lower on 6 of 10 images
  (mean −0.013) while PSNR is higher on all 10. The pooled decode is smoother,
  which SSIM favors on flat backgrounds when a 128 px decode is upscaled.
- **Motion speed.** Clips are sampled uniformly over their length, so a long
  clip stored in few latents plays faster. Kept from upstream.
- **Stacks are pseudo-videos.** Stacked images become consecutive latent
  frames; Text Encode shows them to Qwen as one clip. Splitting them for the
  DiT or for Qwen did not measurably help (see Tried and rejected).
- **Audio.** Input is cut to `max_seconds`, encoded in independent 10 s chunks
  with no silence trimming, and the token budget keeps only the start. Not
  measured here.
- **Core decode.** The one-frame decode loss sits in ComfyUI's H3 VAE path;
  this pack only works around it for its own previews.
- **Existing mods.** 0.2.x mods keep their pooled latents; create them again to
  benefit.
- **Behavior change.** Under `truncate`, a Full Reference that exceeds
  `max_tokens` now shrinks spatially instead of failing or dropping references.
- **Other packs.** ComfyUI-Fantastic-MiniMaxH3-PromptBuilder's
  `refmod_create.py` and `refmod_edit.py` carry copies of the 0.2.x pooling and
  refinement. ComfyUI-H3-Continuity finds the nodes by ID, but its
  `tests/test_refmods.py` hardcodes the `ComfyUI-MiniMaxH3Mod` folder.

## Reproduce

```bash
python custom_nodes/ComfyUI-H3-RefForge/fidelity_bench.py \
    --vae models/vae/minimax-h3/minimax_h3_video_vae_int8_convrot.safetensors \
    --image a.png --image b.jpg --video c.mp4 --out bench.json
```

The defaults match the runs above: grids 8–48, video grids 8 and 16,
`latent_frames` 2, 7 and 16, 500 baseline steps, and the first four images as
the stack case. The new path's scores reproduced exactly in a second run. The
benchmark needs scikit-image for SSIM; the nodes do not.
