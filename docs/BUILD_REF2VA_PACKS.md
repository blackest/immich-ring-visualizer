# Build your own Ref2VA packs (21 GB Q8, 12 GB Q4) on a Mac

The H3 view's "H3 Ref2VA Q8" model (pictures, clips and RefMods as references) needs the **Ref2VA** transformer, which is a
different fine-tune from the FL2VA one most H3 packs carry. The full Q8 build of it is 47.5 GB and needs about 66 GB of RAM
to make. This guide builds a **pruned** pack instead, with a few GB of RAM:

| pack | size | DiT load peak | denoise peak |
|---|---:|---:|---:|
| full Q8 (AdaLN bf16) | 47.5 GB | 19.9 GiB | 22.2 GiB |
| **pruned Q8** (AdaLN float32) | **21 GB** | 19.9 GiB | 22.2 GiB |
| **pruned Q4** (AdaLN float32) | **12 GB** | 10.6 GiB | 12.9 GiB |

Measured on one M2 Max (96 GB), 640x384, 5 s, Fast (3-step), one RefMod, seed 5. They are **smaller and lighter on memory, not
faster** (denoising is compute-bound). The Qwen3-VL-32B text encoder still peaks at 25.7 GiB, which is the RAM floor for plain
renders. Not tested on 36 or 48 GB Macs. Q4 is a visibly different take from Q8 and has about five times the quantization error
in the runner's own measurements (on FL2VA); try it before relying on it.

Everything here is built **on your own Mac from MiniMax's official download**. Nothing is fetched from a third party except
the official weights (and, for the optional Flex step, Pika's adapter at a pinned revision).

## Licence, first

The result is a *MiniMax H3 Model Derivative* under the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). It is not an open-source
licence. The grant is for the Applicable Territory, **worldwide excluding the European Union, the United Kingdom, the
Republic of Korea and the United States**; commercial products above US$20M yearly revenue need MiniMax's written
authorization; the Acceptable Use Policy applies. Build for yourself, and do not re-host the result without the licence text, a
`NOTICE` file, a notice that the files are modified, and "Powered by MiniMax H3". Whether you may use or distribute it where
you are is for you to check; this guide does not say.

## You need

- An Apple Silicon Mac. The prune and quantize steps need only a few GB of RAM and no GPU time.
- **About 130 GB of free disk at the peak**: 66 GB official source + 41 GB pruned intermediate + the pack (21 GB or 12 GB).
  The intermediate can be deleted afterwards, and so can the source if you do not want to rebuild.
- The [minimax-h3-mlx](https://github.com/mrbizarro/minimax-h3-mlx) runner with its virtualenv (`RUNNER` below): Phosphene's
  checkout works, or `vendor/minimax_h3_mlx` in this repository. It needs `mlx`, `numpy` and `safetensors`.
- `hf` (the Hugging Face CLI) to download the official weights.

## 1. Download the official Ref2VA transformer (about 66 GB)

```
hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/transformer/*" --local-dir ~/models/h3-ref2va
```

This gives `~/models/h3-ref2va/Ref2VA/transformer/` with 13 shards. Download time is set by your connection.

## 2. Prune the AdaLN (a few minutes, a few GB of RAM)

```
RUNNER/.venv/bin/python scripts/prune_ref2va_adaln.py --runner RUNNER \
    --src ~/models/h3-ref2va/Ref2VA/transformer \
    --out ~/models/h3-ref2va/pruned/Ref2VA-pruned_bf16.safetensors
```

Every block's AdaLN projection reads `silu(time_embedder(t))`, a nearly one-dimensional curve in 2688 dimensions. The script
re-projects the 51 AdaLN weights onto a 64-dimensional basis, stores a sampled `adaln_t_table` (1001 x 64) and drops the
`time_embedder`. It prints accuracy at the end: expect about 1.7e-6 relative error against the original AdaLN, both at the
grid points and halfway between them. The output is about 41 GB.

## 3. Quantize (about two minutes)

Q8 core:

```
RUNNER/.venv/bin/python scripts/quantize_pruned_stream.py --runner RUNNER \
    --src ~/models/h3-ref2va/pruned/Ref2VA-pruned_bf16.safetensors \
    --out ~/models/h3-ref2va/q8-pruned-f32adaln --bits 8
```

Q4 core: the same with `--bits 4` and a different `--out`. The AdaLN stays in float32 by default (`--adaln-bits 0`). That costs
about 1.3 GB and measured at the bf16 rounding floor, where quantizing it to 8-bit (`--adaln-bits 8`) gave six times the error
on the saved modulation tables; the runner's own notes warn that AdaLN error biases every block at every step.

The script ends with `LOAD OK -- pack valid`, after making `load_dit` build the whole model from the folder.

## 4. Use it

The H3 view finds the Ref2VA pack in this order: the `LTX_H3_REF2VA_DIT` environment variable, `<H3 models folder>/ref2va-q8`,
then `~/models/h3-ref2va/q8`. Point one of those at the folder you built (a symlink is enough), then pick **H3 Ref2VA Q8**.

With the runner directly: `scripts/generate_staged.py --dit <pack>` plus `--refmod` or `--ref2va-reference`. Reference conditioning needs
a runner that has it: the draft PR [mrbizarro/minimax-h3-mlx#2](https://github.com/mrbizarro/minimax-h3-mlx/pull/2), or the
copy in `vendor/minimax_h3_mlx` here.

## Optional: Flex with references

Pika H3 Flex's Ref2VA adapter runs on this trunk. `scripts/build_pika_ref2va_adapter.py --dir <folder>` fetches it at
Phosphene's pinned revision, checks Pika's published SHA-256 and converts it with Phosphene's own `fetch_h3_flex.py`. In our one
test (same seed, one RefMod) it was accepted and the likeness held, in 10 min against Fast's 5.5, so it is not faster.

## Optional: the full 47.5 GB Q8

`scripts/build_ref2va_q8.py` rebuilds the original full pack. It needs about 66 GB of RAM (a 96 GB Mac was barely up to it),
and `--verify <pack>` checks a finished pack's layout without MLX. We rebuilt ours this way and all 10 shards came out
byte-identical to the pack it replaced.

## Notes

- The pruned layout is the one DeepBeepMeep published for FL2VA; the quantizer follows the runner's `quantize_stream.py`
  (CPU stream, bounded memory) with separate core and AdaLN widths.
- Renders are deterministic for a given pack and seed, but a pruned pack is not bit-identical to the full one: the AdaLN tables
  are stored in bf16, and a tiny change there gives a different take from a 3-step sampler (same subject and room, props move).
- Plan notes with all the measurements: `ENHANCED_H3_PLAN.md`, "Update 2026-10-31, later".
