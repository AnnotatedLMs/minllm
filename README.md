  # MinLLM - An Opinionated Guide for Transformer-based LLMs

  ## Overview

  This repository is a very rough reference guide for transformer-based Large Language Models (LLMs). The code is intentionally approximate, not optimized, nor well engineered.

  The organization is very overly granular -- just to make it possible to explain each operation.

  I started writing this while preparing for interviews, and it was intended as a personal reference that I could easily navigate using VSCode navigation features.

  But I provide it here in case anyone else finds this useful.

## Reference Guide
If you're just interested in treating this as a reference guide, you'll still need to follow the installation instructions below to enable the repo navigation.

But if you just want to read the implmenetations, I recommend starting from
`pretraining/common/models/architectures`

From any of the modules, you'll see the architectures at a very high-level, without really diving into the different Transformer block architectures -- as will be apparent from their forward pass methods.

From there, I'd recommend looking at the Transformer Block module they import -- `pretraining/common/models/blocks`. Each block will instantiate the `torch` modules it needs, with mixins that encapsulate how those modules are actually used in the forward pass.

Similarly, when it comes to the different attentnion implementations in `pretraining/common/models/attention`, each attention module will follow this pattern -- instantiating the modules it needs while grouping the forward pass operations into mixins.

Again, all of this is very opinionated organization and is primarily geared towards personal use.

## Installation Guide

### Prerequisites

Install uv (Python package manager):
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

### Quick Setup
Clone the repository and set up the environment:
```bash
# Clone repo
git clone https://github.com/chengyjonathan/minllm.git
cd minllm

# Install dependencies (automatic based on platform):
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Installation Options

#### Automatic Platform Detection
MinLLM automatically installs the appropriate PyTorch version based on your platform:
- **macOS/Windows**: CPU-only PyTorch builds
- **Linux**: CUDA 12.8 PyTorch builds (for GPU support)

Simply run:
```bash
uv sync
```

#### Manual Override (if needed)
If you need to override the automatic selection, you can modify the markers in `pyproject.toml` or manually install PyTorch after setup.

##### Optional: FlashMLA Installation (Linux + NVIDIA GPU only)
For optimized DeepSeek3 attention on compatible GPUs (SM90/SM100):
```bash
# After running uv sync
git clone https://github.com/deepseek-ai/FlashMLA.git
cd FlashMLA
uv pip install -v .
cd ..
```
**Note:** FlashMLA requires NVIDIA GPU with SM90/SM100 architecture (Ada Lovelace/Hopper), CUDA 12.x, and Linux. It will be automatically detected and used if available.

### Optional Setup

```bash
# Install only production dependencies (no dev tools)
uv sync --no-dev
```

## Data Setup
- Borrowed a script from KellerJordan/modded-nanogpt
- Download a subset of fineweb to use with the pretraining scripts

```
# downloads only the first 800M training tokens to save time
uv run pretraining.data.sample.cached_fineweb 8
```

## Implementation Legend
✅ Means I have implemented it, and I'm able to do runs with it
⚠️ Means I have implmeneted it, and I'm planning to test it in the nearish future
❌ Means I want to implement it, and I'd like to get to it at some point.

## Supported Models

| Model | CPU | GPU | DDP | FSDP |
|-------|-----|-----|-----|------|
| GPT-2 | ✅ | ⚠️ | ⚠️ | ⚠️ |
| LLaMA 3.1 | ✅ | ⚠️ | ⚠️ | ⚠️ |
| DeepSeek 3 | ✅ | ⚠️ | ⚠️ | ⚠️ |
| Qwen 3 | ❌ | ❌ | ❌ | ❌ |
| Kimi k2 | ❌ | ❌ | ❌ | ❌ |

## Supported Post-training

| Method | Status |
|--------|--------|
| YaRN | ❌ |
| PPO | ❌ |
| DPO | ❌ |
| GRPO | ❌ |
| GSPO | ❌ |

## Contributing Guidelines

Please, feel free to make PRs and changes. I'm just a dummy trying to make an educational resource - without a lot of the bells and whistles.

## License

MIT License

## Acknowledgements

  This repository was heavily inspired by:

  - https://github.com/karpathy/nanoGPT
  - https://github.com/karpathy/nano-llama31
  - https://github.com/allenai/OLMo
  - https://github.com/allenai/open-instruct
  - https://github.com/KellerJordan/modded-nanogpt

Really, a mix of Kaparthy, AI2, Raschka. But instead of hopping around a bunch of different guides, I just really wanted a centralized place where I could see all of the stuff together. I reorganized a lot of their code, annotated stuff, and invariably made it worse to use -- but easier (for me) to read.

## Understanding pyproject.toml

When you run `uv sync`, here's how the configuration automatically selects the right PyTorch version:

### 1. Base Dependencies (always installed)
```toml
[project]
dependencies = [
    "torch>=2.7.1",          # Always installed (source depends on platform)
    "accelerate>=1.8.1",
    "datasets>=3.6.0",
    # ... etc
]
```

### 2. Platform-based Source Resolution
```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", marker = "sys_platform != 'linux'" },   # macOS/Windows
  { index = "pytorch-cu128", marker = "sys_platform == 'linux'" }, # Linux (GPU)
]
```
**Note:** The `marker` conditions automatically select the appropriate PyTorch build based on your operating system:
- Non-Linux systems (macOS, Windows) get CPU-only builds
- Linux systems get CUDA 12.8 builds for GPU support

### 3. Index Definitions (PyTorch wheel locations)
```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

### Full Flow
- `uv sync` on macOS/Windows: Automatically installs torch from CPU index
- `uv sync` on Linux: Automatically installs torch from CUDA 12.8 index
- No need to specify extras or manually select versions
- FlashMLA: Must be manually installed for GPU users (see Installation Options)
