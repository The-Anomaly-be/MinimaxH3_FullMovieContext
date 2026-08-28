#!/usr/bin/env python3
"""
render.py - MiniMax H3 T2V / REF2V 15s via Motion Context chain, driven by a text file.

Each PARAGRAPH (block separated by one or more blank lines) in prompt.txt = ONE 15s video
(3 clips of 5s chained through MotionContext for continuity).

Usage:
  python render.py prompt.txt
  python render.py prompt.txt --host http://127.0.0.1:8192 --out output
  python render.py prompt.txt --ref2v image.png        # image-to-video (REF2V)
  python render.py prompt.txt --clips 4                 # 4 clips x 5s = 20s
  python render.py prompt.txt --width 768 --height 432 # lower resolution

Requirements (see README.md):
  - ComfyUI running on --host, with custom nodes: ComfyUI-GGUF, ComfyUI-KJNodes,
    ComfyUI-Spectrum-MiniMax-H3 (optional), ComfyUI-H3-Motion-Context.
  - GGUF models loaded (FL2VA / REF2VA unet, qwen3vl clip, video+audio VAE).
  - TurboLoRA / SLA (PlagueKind) DISABLED: they break the H3 forward pass
    (SLA needs triton, Linux-only; TurboLoRA merge blocks on low RAM).

Note: MiniMaxH3ImageToVideo does not yet wire first_frame/last_frame in this build,
so REF2V uses the same T2V node (prompt drives it; the image is not yet chained in).
"""
import json, urllib.request, urllib.error, sys, time, os, glob, argparse, subprocess

HOST = "http://127.0.0.1:8192"
UNET_T2V = "MiniMax-H3-FL2VA-Q3_K_M.gguf"
UNET_R2V = "MiniMax-H3-REF2VA-Q3_K_M.gguf"
CLIP_GGUF = "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

WIDTH, HEIGHT = 1024, 576
CLIPS = 3            # 3 x 5s = 15s
LENGTH = 124        # frames per 5s clip @ 24fps

def split_paragraphs(text):
    """Split text into paragraphs (separated by >=1 blank line)."""
    paras, buf = [], []
    for line in text.splitlines():
        if line.strip() == "":
            if buf:
                paras.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        paras.append("\n".join(buf).strip())
    return [p for p in paras if p]

def build_clip(clip_idx, latent_path, prompt, width, height, ref_image=None):
    """clip_idx: 1..N. latent_path: None for first clip.
    ref_image: optional start image (REF2V) - wired when the node supports it."""
    p = {}
    unet = UNET_R2V if ref_image else UNET_T2V
    p["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}}
    p["2"] = {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": CLIP_GGUF, "type": "wan"}}
    p["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}}
    p["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}}
    itv = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
           "width": width, "height": height, "length": LENGTH}
    p["5"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": itv}
    p["267"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
        "model": ["1", 0], "shift_video": 12.0, "shift_audio": 6.0}}
    p["270"] = {"class_type": "MiniMaxLowVRAMAttention", "inputs": {
        "model": ["267", 0], "head_chunks": 4}}
    if latent_path is None:
        conditioning_src = ["5", 0]
        sampler_latent = ["5", 1]
    else:
        p["178"] = {"class_type": "MiniMaxH3MotionContextLoadLatent", "inputs": {
            "latent_path": latent_path, "clip_index": clip_idx - 1}}
        p["240"] = {"class_type": "MiniMaxH3MotionContext", "inputs": {
            "conditioning": ["5", 0], "vae": ["3", 0], "latent": ["5", 1],
            "context_latent": ["178", 0], "context_length": "22",
            "audio_context_length": 24}}
        conditioning_src = ["240", 0]
        sampler_latent = ["5", 1]
    p["7"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 1234 + clip_idx}}
    p["8"] = {"class_type": "BasicScheduler", "inputs": {
        "model": ["270", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}}
    p["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    p["10"] = {"class_type": "BasicGuider", "inputs": {
        "model": ["270", 0], "conditioning": conditioning_src}}
    p["11"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["7", 0], "guider": ["10", 0], "sampler": ["9", 0],
        "sigmas": ["8", 0], "latent_image": sampler_latent}}
    p["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}
    p["13"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}}
    p["179"] = {"class_type": "MiniMaxH3MotionContextSaveLatent", "inputs": {
        "latent": ["11", 0], "filename_prefix": "H3_mc", "clip_index": clip_idx}}
    p["14"] = {"class_type": "CreateVideo", "inputs": {
        "images": ["12", 0], "audio": ["13", 0], "fps": 24}}
    tag = "R2V" if ref_image else "T2V"
    p["15"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["14", 0], "filename_prefix": "H3_%s_mc_clip%d" % (tag, clip_idx), "format": "mp4"}}
    return p

def post(payload, host):
    data = json.dumps({"prompt": payload, "client_id": "h3_render"}).encode()
    req = urllib.request.Request(host + "/api/prompt",
        data=data, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

def find_latest_latent():
    matches = sorted(glob.glob(os.path.join("output", "H3_mc_*.safetensors")),
                     key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None

def wait_done(pid, host, timeout_min=45):
    t0 = time.time()
    for _ in range(timeout_min * 20):
        try:
            h = json.loads(urllib.request.urlopen(host + "/history/" + pid, timeout=30).read())
        except Exception:
            h = {}
        if pid in h:
            out = h[pid].get("outputs", {})
            print("  DONE in %.1f min." % ((time.time()-t0)/60))
            for nid, o in out.items():
                for f in o.get("videos", []) + o.get("gifs", []) + o.get("audio", []):
                    print("    FILE[%s]:" % nid, f.get("filename"))
            return True, find_latest_latent()
        time.sleep(3)
    print("  TIMEOUT")
    return False, None

def concat_clips(out_dir, prefix, n, out_name):
    clips = []
    for i in range(1, n + 1):
        m = sorted(glob.glob(os.path.join(out_dir, "%s%d_*.mp4" % (prefix, i))))
        if not m:
            print("  ERROR: clip %d missing" % i); return None
        clips.append(m[-1])
    list_path = os.path.join(out_dir, "_h3_concat_list.txt")
    with open(list_path, "w") as f:
        for c in clips:
            f.write("file '%s'\n" % c.replace("\\", "/"))
    out_path = os.path.join(out_dir, out_name)
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path],
                       capture_output=True, text=True)
    try: os.remove(list_path)
    except: pass
    if r.returncode != 0:
        print("  FFMPEG ERROR:", r.stderr[-800:]); return None
    return out_path

def render_prompt(prompt, args, idx_total):
    tag = "R2V" if args.ref2v else "T2V"
    prev = None
    for i in range(1, args.clips + 1):
        print("  [%s clip %d/%d]" % (tag, i, args.clips))
        p = build_clip(i, prev, prompt, args.width, args.height, ref_image=args.ref2v)
        try:
            r = post(p, args.host)
        except urllib.error.HTTPError as e:
            print("  HTTP ERROR", e.code, e.read().decode()[:400]); return False
        pid = r["prompt_id"]
        ok, prev = wait_done(pid, args.host)
        if not ok:
            print("  Stopped (clip failed)"); return False
    prefix = "H3_%s_mc_clip" % tag
    secs = args.clips * 5
    out_name = "H3_%s_%ds_MC_%05d_.mp4" % (tag, secs, idx_total)
    out = concat_clips(args.out, prefix, args.clips, out_name)
    if out:
        print("  -> %s" % out)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt_file")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--out", default="output")
    ap.add_argument("--ref2v", default=None, help="start image (REF2V)")
    ap.add_argument("--clips", type=int, default=CLIPS)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    text = open(args.prompt_file, encoding="utf-8").read()
    paras = split_paragraphs(text)
    if not paras:
        print("No paragraph found in", args.prompt_file); sys.exit(1)
    print("%d prompt(s) to render (%d clips x 5s each)." % (len(paras), args.clips))
    for k, para in enumerate(paras, 1):
        head = para[:200] + ("..." if len(para) > 200 else "")
        print("\n=== PROMPT %d/%d ===\n%s" % (k, len(paras), head))
        if not render_prompt(para, args, k):
            print("Failed prompt", k); sys.exit(1)
    print("\nDone.")

if __name__ == "__main__":
    main()
