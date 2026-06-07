# ComfyUI_JoyAI_Echo

ComfyUI nodes for [JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo) — minute-level multi-shot audio-video generation with paired cross-modal memory.

This implementation is **faithful to the official inference pipeline** with zero precision loss. Key differences from existing community implementations:

- **Full bf16 precision** for text encoder (Gemma-3-12b) — no GGUF quantization
- **Correct parameters** matching official defaults (1280x736, 241 frames, 25fps)
- **GPU memory optimization** via module hot-swap (no quality degradation)
- **Built-in LLM prompt enhancement** via cloud API (zero local VRAM usage)

## Requirements

- NVIDIA GPU with **48GB+ VRAM** (A6000/H100/A100 80GB recommended)
  - With hot-swap enabled: peak ~46GB during denoise phase
  - With **sequential offload** enabled: **24GB VRAM** sufficient (slower inference)
- Python 3.11+
- PyTorch 2.4+ with CUDA support
- ffmpeg (for video concatenation)

## Installation

### 1. Clone this repo into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zhuang2002/ComfyUI_JoyAI_Echo.git
```

### 2. Install dependencies

```bash
cd ComfyUI_JoyAI_Echo
pip install -r requirements.txt
```

Or let ComfyUI auto-install via `install.py` on first launch.

### 3. Download model weights

| File | Size | Source |
|------|------|--------|
| `JoyAI-Echo-release.safetensors` | ~46 GB | [HuggingFace](https://huggingface.co/jdopensource/JoyAI-Echo) |
| `gemma-3-12b-it/` (full bf16) | ~24 GB | [HuggingFace](https://huggingface.co/google/gemma-3-12b-it) |

Place them anywhere accessible. You'll provide paths in the nodes.

## Nodes

### JoyEcho Model Loader

Loads all model components (text encoder, DiT generator, VAEs).

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| checkpoint_path | STRING | — | Path to `JoyAI-Echo-release.safetensors` |
| gemma_path | STRING | — | Path to `gemma-3-12b-it` directory |
| lora_path | STRING | "" | Optional LoRA weights for memory conditioning |
| lora_strength | FLOAT | 1.0 | LoRA strength |
| low_vram | BOOLEAN | False | Load text encoder on CPU (saves ~24GB VRAM) |

### JoyEcho Text Encode

Encodes text prompts using Gemma-3-12b. Supports multiple input formats:

- **One prompt per line** — each line is one shot
- **JSON object** — `{"prompts": ["shot1", "shot2", ...]}`
- **JSON file path** — path to a `.json` file (absolute or relative to the node directory)

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| model | JOYECHO_MODEL | — | From Model Loader |
| prompts | STRING | — | Multi-line text, JSON, or `.json` file path |
| release_text_encoder | BOOLEAN | True | Free ~24GB VRAM after encoding |

### JoyEcho Generate (Multi-Shot)

Runs the full inference pipeline with **paired audio-video memory bank**. Each shot is conditioned on previous shots through visual identity and voice timbre memory, maintaining story-level consistency.

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
| audio_memory_window_size | INT | 96 | Audio memory window size |
| sequential_offload | BOOLEAN | False | Layer-by-layer offloading (24GB VRAM, slower) |

**Outputs**: IMAGE (all frames concatenated) + AUDIO (combined waveform)

### JoyEcho LLM Enhance

Calls a cloud LLM API to automatically expand a short story idea into properly formatted shot prompts. **Uses zero local GPU memory** — only cloud API calls.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| story_idea | STRING | — | Your story or scene idea in a few sentences |
| mode | ENUM | long_story | `long_story (multi-shot)` or `short_story (single-shot)` |
| api_key | STRING | — | Your API key (OpenAI, DeepSeek, etc.) |
| base_url | STRING | `https://api.openai.com/v1` | API base URL |
| model_name | STRING | gpt-4o | LLM model name |
| num_shots | INT | 0 | Number of shots (0 = let LLM decide) |
| temperature | FLOAT | 0.7 | Sampling temperature |

Supported API providers (any OpenAI-compatible API):
- **OpenAI**: base_url = `https://api.openai.com/v1`, model = `gpt-4o`
- **DeepSeek**: base_url = `https://api.deepseek.com/v1`, model = `deepseek-chat`
- **Other**: Any OpenAI-compatible endpoint

### JoyEcho Prompt Format (Helper)

Outputs the official system prompt for LLM-based prompt writing. Use this with external LLM nodes if you prefer to handle the API call yourself.

## Workflows

Two ready-to-use workflows are included in the `workflows/` directory:

### Basic Workflow (`joyai_echo_basic.json`)

```
[Model Loader] → [Text Encode] → [Generate] → [Preview Image]
                                              → [Preview Audio]
```

Loads a prompt JSON file and generates multi-shot video with memory. The Text Encode node is pre-configured with `prompts/test_001.json` (15-shot vlog story).

### LLM Enhanced Workflow (`joyai_echo_llm_enhanced.json`)

```
[LLM Enhance] → [Text Encode] → [Generate] → [Preview Image]
[Model Loader] ↗                             → [Preview Audio]
```

Enter a story idea in natural language → LLM generates shot prompts → model generates video with paired memory.

## Example Prompts

8 test prompt files are included in `prompts/`:

| File | Shots | Description |
|------|-------|-------------|
| test_001.json | 15 | Young woman recording evening vlog |
| test_002.json | 15 | Cafe reflection story |
| test_003.json | 15 | Park walk narrative |
| test_004.json | 15 | Kitchen cooking scene |
| test_005.json | 15 | Library study session |
| test_006.json | 15 | Morning routine |
| test_007.json | 15 | Art studio session |
| test_008.json | 15 | Rainy day at home |

System prompts for LLM-based prompt writing:
- `prompts/long_story_writer_system_prompt.md` — multi-shot story generation
- `prompts/short_story_writer_system_prompt.md` — single-shot scene generation

## Memory Management

The node uses the same hot-swap strategy as the official code:

1. **Text encoding phase**: Text encoder on GPU (~24GB), everything else off
2. **Denoise phase**: DiT generator on GPU (~30GB), VAEs on CPU
3. **Decode phase**: Generator moved to CPU, VAE decoders + vocoder on GPU

This ensures peak VRAM never exceeds ~48GB despite the model totaling ~70GB+ in parameters.

### Paired Audio-Video Memory Bank

For multi-shot generation, the memory bank maintains:
- **Video memory**: Key frames from previous shots for visual identity consistency
- **Audio memory**: Voice timbre and audio characteristics for audio consistency
- Memory slots are managed with configurable max size and fixed slots

### Sequential Offload Mode (24GB GPUs)

Enable `sequential_offload` in the Generate node to run on 24GB cards (e.g. RTX 4090, 3090):

- Only one transformer block (~600MB) is loaded to GPU at a time
- Non-block layers (~2GB) stay on GPU permanently
- Blocks use pinned CPU memory for fast transfers
- **Trade-off**: ~3-5x slower inference per shot
- **Zero precision loss** — identical output to full-VRAM mode

## Differences from Official Pipeline

This node produces **identical results** to `python inference.py` when using the same parameters:
- Same denoising schedule (8 steps, DMD distilled sigmas)
- Same flow-matching noise formula
- Same paired audio-video memory bank with max_response window selection
- Same RoPE position encoding

## Acknowledgements

- [JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo) by Echo Team @ Joy Future Academy, JD
- [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) by Lightricks
- [Gemma-3](https://huggingface.co/google/gemma-3-12b-it) by Google

## License

For academic research and non-commercial use only (following the upstream JoyAI-Echo license).
