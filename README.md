# ComfyUI_JoyAI_Echo

ComfyUI nodes for [JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo) — minute-level multi-shot audio-video generation with paired cross-modal memory.

This implementation is **faithful to the official inference pipeline** with zero precision loss. Key differences from existing community implementations:

- **Full bf16 precision** for text encoder (Gemma-3-12b) — no GGUF quantization
- **Correct parameters** matching official defaults (1280x736, 241 frames, 25fps)
- **GPU memory optimization** via module hot-swap (no quality degradation)

## Requirements

- NVIDIA GPU with **48GB+ VRAM** (A6000/H100/A100 80GB recommended)
  - With hot-swap enabled: peak ~46GB during denoise phase
- Python 3.11+
- PyTorch 2.4+ with CUDA support
- ffmpeg (for video concatenation)

## Installation

### 1. Clone this repo into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/YOUR_USERNAME/ComfyUI_JoyAI_Echo.git
```

The `--recursive` flag pulls the official JoyAI-Echo repo as a git submodule.

### 2. Install dependencies

```bash
cd ComfyUI_JoyAI_Echo
pip install -r requirements.txt
```

Or let ComfyUI auto-install via `install.py` on first launch.

### 3. Download model weights

| File | Size | Source |
|------|------|--------|
| `echo-longvideo-release.safetensors` | ~46 GB | [HuggingFace](https://huggingface.co/jdopensource/JoyAI-Echo) |
| `gemma-3-12b-it/` (full bf16) | ~24 GB | [HuggingFace](https://huggingface.co/google/gemma-3-12b-it) |

Place them anywhere accessible. You'll provide paths in the nodes.

## Nodes

### JoyEcho Model Loader

Loads all model components (text encoder, DiT generator, VAEs).

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| checkpoint_path | STRING | — | Path to `echo-longvideo-release.safetensors` |
| gemma_path | STRING | — | Path to `gemma-3-12b-it` directory |
| lora_path | STRING | "" | Optional LoRA weights for memory conditioning |
| lora_strength | FLOAT | 1.0 | LoRA strength |

### JoyEcho Text Encode

Encodes text prompts using Gemma-3-12b. One prompt per line = one shot.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| model | JOYECHO_MODEL | — | From Model Loader |
| prompts | STRING | — | Multi-line text, one shot per line |
| release_text_encoder | BOOLEAN | True | Free ~24GB VRAM after encoding |

### JoyEcho Generate (Multi-Shot)

Runs the full inference pipeline with paired audio-video memory bank.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| model | JOYECHO_MODEL | — | From Model Loader |
| conditioning | JOYECHO_COND | — | From Text Encode |
| seed | INT | 12345 | Random seed (increments per shot) |
| num_frames | INT | 241 | Frames per shot (must be 1+8k) |
| video_height | INT | 736 | Video height in pixels |
| video_width | INT | 1280 | Video width in pixels |
| video_fps | INT | 25 | Video frame rate |
| v2a_grad_scale | FLOAT | 2.0 | Video-to-audio cross-modal scale |
| memory_max_size | INT | 7 | Max memory bank slots |
| num_fix_frames | INT | 3 | Fixed (non-evictable) memory slots |
| enable_audio_memory | BOOLEAN | True | Enable audio memory conditioning |

**Outputs**: IMAGE (all frames concatenated) + AUDIO (combined waveform)

## Memory Management

The node uses the same hot-swap strategy as the official code:

1. **Text encoding phase**: Text encoder on GPU (~24GB), everything else off
2. **Denoise phase**: DiT generator on GPU (~30GB), VAEs on CPU
3. **Decode phase**: Generator moved to CPU, VAE decoders + vocoder on GPU

This ensures peak VRAM never exceeds ~48GB despite the model totaling ~70GB+ in parameters.

## Workflow Example

```
[JoyEcho Model Loader] → [JoyEcho Text Encode] → [JoyEcho Generate] → [Preview Image]
                                                                     → [Preview Audio]
```

For multi-shot stories, enter multiple lines in the Text Encode node:

```
A young woman with long black hair sits at a cafe table, speaking softly...
The same woman walks through a rainy street, her voice narrating...
She arrives at a bookstore, the bell chiming as she enters...
```

Each line generates one shot. The memory bank automatically maintains visual and audio consistency across shots.

## Differences from Official Pipeline

This node produces **identical results** to `python inference.py` when using the same parameters:
- Same denoising schedule (8 steps, DMD distilled sigmas)
- Same flow-matching noise formula: `x_t = (1 - sigma) * x_0 + sigma * noise`
- Same stochastic re-corruption at each step
- Same paired audio-video memory bank with max_response window selection
- Same RoPE position encoding (VIDEO_FPS=24.0 for temporal positions)

## Acknowledgements

- [JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo) by Echo Team @ Joy Future Academy, JD
- [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) by Lightricks
- [Gemma-3](https://huggingface.co/google/gemma-3-12b-it) by Google

## License

For academic research and non-commercial use only (following the upstream JoyAI-Echo license).
