# Render evaluation

Question: does H3 generate a more faithful person when a RefMod is presented
differently, and can the same likeness cost fewer tokens? [AUDIT.md](AUDIT.md)
scores the stored latent; this scores the generated video. The baseline is
RefForge 0.3.0 (`a9d8809`). One change passed and ships in 0.3.1: Text Encode
shows a stack to Qwen at no more than 54 vision tokens per frame.

## Setup

- **References.** Six stills of one real person (two close-ups, three medium
  shots, one full-length shot) stacked into one Compressed Reference by the
  0.3.0 Create node: grid 36 (36×24 cells, 1296 DiT tokens) and grid 24 (24×16
  cells, 576 tokens). Creating them again with the tested build gave
  bit-identical latents.
- **Renders.** H3 ref2va and Qwen3-VL text encoder (both int8), 512×768, 73
  frames, 8 Euler steps, no CFG. Two prompts in H3's section format, a portrait
  and a full-body shot, with three seeds each; the grid-36 rows with n = 10
  have five. The plain subject definition is `<Subject 1> is the
  same adult woman shown in all the views in <Video 1>.`; the described one is
  `<Subject 1> is an adult woman with long brown hair and brown eyes, the same
  woman shown in all the views in <Video 1>.`
- **Identity.** SFace (OpenCV zoo) cosine between the largest face in every
  4th frame and each source still, averaged over the six. OpenCV's same-person
  threshold is 0.363. Without references the renders score 0.22, with the
  person's name 0.26, with the grid-36 RefMod 0.62.
- **Comparison.** Differences are paired on prompt and seed; t = mean / (sd /
  √n). For n = 6, |t| > 2.57 means p < 0.05; for n = 10, |t| > 2.26. Rendering
  is deterministic: 12 renders repeated in a new ComfyUI session reproduced
  their scores exactly, so a paired difference comes from the conditioning.
- **Tokens.** DiT reference tokens plus the Qwen vision tokens of what Text
  Encode shows the encoder. Both are in the DiT sequence at every step.

## Shipped in 0.3.1

Text Encode area-downscales a stack's view to at most 54 Qwen tokens per frame
(192×288 px at 2:3). At grid 36 that is exactly the half-size view in the
results below; the default grid-16 view (40 tokens) is unchanged. The rule was
fixed before the final renders: both prompts at least −0.005 and neither
significantly negative, the pooled 95% lower bound at least −0.015, and grid 24
not significantly negative.

| Compared with 0.3.0 | Tokens | Identity Δ (95% CI) | t | n |
| --- | ---: | --- | ---: | ---: |
| Plain prompt, grid 36 | 1512 → 1350 | +0.005 (−0.012, +0.021) | 0.63 | 10 |
| Appearance described, grid 36 | 1512 → 1350 | +0.007 (−0.006, +0.020) | 1.21 | 10 |
| Both prompts, grid 36 | 1512 → 1350 | +0.006 (−0.004, +0.015) | 1.27 | 20 |
| Plain prompt, grid 24 | 672 → 630 | −0.003 (−0.019, +0.014) | −0.44 | 6 |

Renders were 1.9 s faster at grid 36 (paired over 20, t = −9.2) and 1.4 s
faster at grid 24 (t = −8.1).

## Results

0.3.0 Text Encode decodes a stack as one clip and samples it at 2 fps, so Qwen
sees one vision block of two frames: still 1, and a frame that blends still 4
with its neighbours (16 dB against still 4). Rows marked RefForge build and
the paired view ran through experimental builds of RefForge's Text Encode, and
0.3.1 rows through the shipped one. The other per-still rows ran through
Fantastic's Text Encode, which shows stills stored at the given size as
encoder frames.

Plain prompt, compared with 0.3.0 at grid 36:

| Presentation | DiT + Qwen tokens | Identity portrait / body | Δ | t | n | s/render |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0.3.0 (baseline) | 1296 + 216 | 0.694 / 0.554 | | | 10 | 38.2 |
| Appearance in the subject definition | 1296 + 216 | 0.699 / 0.587 | +0.019 | 2.48 | 10 | 37.6 |
| 0.3.1: the 0.3.0 view at half size | 1296 + 54 | 0.697 / 0.560 | +0.005 | 0.63 | 10 | 36.1 |
| A block per still, 24 tokens each (Fantastic) | 1296 + 144 | 0.707 / 0.557 | +0.018 | 2.72 | 6 | 38.1 |
| A block per still, 24 tokens each (RefForge build) | 1296 + 144 | 0.710 / 0.550 | +0.006 | 1.06 | 10 | 40.4 |
| A block per still, 54 tokens each | 1296 + 324 | 0.702 / 0.543 | +0.008 | 1.57 | 6 | 39.4 |
| A block per still, 216 tokens each | 1296 + 1296 | 0.709 / 0.566 | +0.023 | 2.91 | 6 | 45.1 |
| Stills paired into three blocks | 1296 + 648 | 0.710 / 0.537 | +0.009 | 1.34 | 6 | 41.9 |
| Six source pictures (core node, 384×576) | 1296 + 1296 | 0.696 / 0.567 | +0.017 | 1.94 | 6 | 47.7 |
| Grid 24, 0.3.0 | 576 + 96 | 0.637 / 0.549 | −0.022 | −1.33 | 6 | 34.4 |
| Grid 24, 0.3.1 | 576 + 54 | 0.633 / 0.547 | −0.024 | −1.48 | 6 | 33.0 |
| Grid 24, a block per still, 24 tokens (RefForge build) | 576 + 144 | 0.647 / 0.535 | −0.024 | −1.72 | 6 | 36.3 |
| Grid 24, a block per still, 96 tokens | 576 + 576 | 0.669 / 0.540 | −0.010 | −1.26 | 6 | 37.6 |

Appearance in the subject definition, compared with 0.3.0 at grid 36 under the
same prompt:

| Presentation | DiT + Qwen tokens | Δ | t | n |
| --- | ---: | ---: | ---: | ---: |
| 0.3.1: the 0.3.0 view at half size | 1296 + 54 | +0.007 | 1.21 | 10 |
| A block per still, 24 tokens each (RefForge build) | 1296 + 144 | −0.005 | −0.47 | 10 |
| A block per still, 54 tokens each | 1296 + 324 | +0.016 | 1.18 | 6 |
| A block per still, 216 tokens each | 1296 + 1296 | +0.015 | 1.40 | 6 |
| Apply only: the DiT gets the RefMod, the prompt names no reference | 1296 + 0 | +0.012 | 0.76 | 6 |
| Apply only, six picture blocks instead of one clip (vs the row above) | 1296 + 0 | −0.018 | −0.64 | 6 |
| Grid 24, 0.3.0 | 576 + 96 | −0.042 | −4.51 | 6 |
| Grid 24, a block per still, 96 tokens | 576 + 576 | −0.027 | −2.27 | 6 |

## Findings

- **Describe the subject.** Adding the appearance ("an adult woman with long
  brown hair and brown eyes") to the subject definition was the only robust
  gain at no token cost.
- **Qwen's view of a stack barely matters.** With the plain prompt every
  per-still view scored above the 0.3.0 view, but only the full-size one
  clearly, at 1080 more tokens and 7 s more per render. The 24-token view built
  into RefForge gained +0.006 (n = 10), not the +0.018 measured through
  Fantastic. Both use the same latent, prompt and thumbnail size; they differ in
  how the thumbnails were decoded and stored. Small conditioning changes move
  identity by 0.01–0.02 per render at a fixed seed, which is the resolution of
  this test. With the appearance described, no view gained significantly. The
  per-still build was rejected: no measured gain, 2.2 s slower, and for a
  six-still stack more Qwen tokens than 0.3.0 at grids below 30. Shrinking the
  0.3.0 view instead kept the likeness and ships in 0.3.1.
- **One clip for the DiT.** Splitting a stack into six picture blocks did not
  help (−0.018 at grid 36, −0.020 at grid 24, both with Apply only).
- **DiT tokens carry the likeness.** Grid 24 scored 0.022–0.042 below grid 36.
  The best grid-24 Qwen view, 96 tokens per still, still scored 0.010–0.027
  lower with 576 + 576 tokens, and Apply alone was not measurably worse than
  Text Encode once the appearance was described.
- **Creation is unchanged.** Through a real VAE round trip, area downscaling
  kept the most facial identity (SFace of the decode against the source: 0.851
  at grid 36, 0.779 at grid 24). Lanczos, antialiased bicubic and an unsharp
  mask scored 0.002–0.019 lower, consistent with AUDIT.md.

## Limits

One person, stacks of stills, two prompts, 6–10 paired renders per row: the
paired sd is 0.01–0.04, so differences under about 0.015 are not resolved.
SFace measures face identity only, not body, hair, clothing or style. Single
images, videos, Full References, bundles and other grids were not rendered.
Times compare ComfyUI sessions on the same machine; the 0.3.0 renders ran in an
earlier one. The harness is not included: it needs the private source stills
and the YuNet and SFace models.
