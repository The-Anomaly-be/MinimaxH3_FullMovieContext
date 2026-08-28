# h3_render — MiniMax H3 T2V / REF2V 15s (Motion Context chain)

A headless ComfyUI driver to render **15s** (or longer) MiniMax H3 videos by chaining
**5s clips through the Motion Context node** (motion + audio continuity). A `prompt.txt`
file = as many videos as it has paragraphs.

## Prepare the ground (REQUIRED)

This script only drives an already-running ComfyUI server. You must first set up:

### 1. ComfyUI
A recent ComfyUI install (tested on 0.33.x), launched with:
```
python main.py --listen 127.0.0.1 --port 8192
```
(the `--listen` + port must match the script's `--host`, default `http://127.0.0.1:8192`)

### 2. Required custom nodes
In `ComfyUI/custom_nodes/`:
- **ComfyUI-GGUF** — loads `.gguf` models (UNET + CLIP)
- **ComfyUI-KJNodes** — provides `MiniMaxLowVRAMAttention` (needed on low RAM)
- **ComfyUI-H3-Motion-Context** — the `MiniMaxH3MotionContext` / `SaveLatent` / `LoadLatent` nodes
- *(optional)* **ComfyUI-Spectrum-MiniMax-H3** — for Spectrum upscale (not used by default here)

> ⚠️ **Do NOT enable** (they break the H3 forward pass on Windows / low RAM):
> - `ComfyUI-MiniMax-H3-Turbo` (TurboLoRA) — the LoRA merge blocks on RAM < 64 GB
> - `ComfyUI-PlagueKind-Nodes` (contains `H3SLAAttention`) — needs **triton**, which has
>   **no Windows wheel** (Linux-only). It fails silently and breaks sampling.
> To try them, rename their folders to `.disabled` to neutralize them.

### 3. Models (place in the usual ComfyUI folders)
| File | Folder | Note |
|---|---|---|
| `MiniMax-H3-FL2VA-Q3_K_M.gguf` | `models/unet/` | T2V (text→video). Q3_K_M fits ~32 GB RAM. Below that, no 15s. |
| `MiniMax-H3-REF2VA-Q3_K_M.gguf` | `models/unet/` | REF2V (image→video), if using `--ref2v` |
| `qwen3vl-32B-MiniMax-H3-Q2_K.gguf` | `models/text_encoders/` | Text CLIP (Q2_K lightweight) |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | Video VAE |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | Audio VAE |

GGUF source: HuggingFace repo `realrebelai/MiniMax-H3_GGUFs`
(FL2VA / REF2VA in `Q3_K_M` and `Q4_K_M`; CLIP in `Q2_K`).

## Usage

```
python render.py prompt.txt
python render.py prompt.txt --host http://127.0.0.1:8192 --out output
python render.py prompt.txt --clips 4            # 4 x 5s = 20s
python render.py prompt.txt --width 768 --height 432   # lower res (tight RAM)
python render.py prompt.txt --ref2v image.png   # REF2V (start image)
```

Or just double-click `example.bat` (runs `examples/prompt.txt`).

### prompt.txt format
**One paragraph = one video.** Paragraphs are separated by **one or more blank lines**.
No special separator, no scene folders — just a plain text file.

```
A small red robot on a rainy neon street at night, looking up at the camera.
Audio: soft rainfall, distant city hum.

A slow wave crashing on rocks, cinematic.
Audio: ocean wash, wind.
```

→ renders 2 videos of 15s each (3 chained 5s clips per video).

## How it works
Each video = N 5s clips (default 3 → 15s):
- **Clip 1**: native T2V (no context).
- **Following clips**: `MiniMaxH3MotionContext` with `context_latent` = latent saved by the
  previous clip (`SaveLatent` / `LoadLatent`), `context_length=22`, `audio_context_length=24`.
  This is the continuity (frames don't "reboot" between clips).
- Clips are then concatenated into a single MP4 (ffmpeg).

Settings: `SigmaShift(12/6)`, `LowVRAMAttention(head_chunks=4)`, euler/simple, 20 steps, 1024×576.

## Example included
`examples/prompt.txt` (1 paragraph = the robot prompt) and
`examples/H3_T2V_15s_MC_00001_.mp4` (the corresponding generated 15s video).

## Notes / limits
- **REF2V**: `MiniMaxH3ImageToVideo` does not yet wire `first_frame`/`last_frame` in this
  build — `--ref2v` therefore uses the same T2V node (prompt drives it). Enable once the
  node exposes the start image.
- **Duration**: one 5s clip takes ~25 min on an RTX 3070 Laptop (8 GB VRAM) / 32 GB RAM.
  A 15s video = ~75 min. Plan accordingly.
- **Seam micro-stutter**: a crossfade (ffmpeg) can smooth it further if needed; the Motion
  Context already smooths a lot, but clip N+1 restarts its own noise.
