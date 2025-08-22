  # MinLLM - An Opinionated Guide for Transformer-based LLMs

  ## Overview

  This repository is a very rough reference guide for transformer-based Large Language Models (LLMs). The code is intentionally approximate, not optimized, nor well engineered.

  The organization is very overly granular -- just to make it possible to explain each operation.

  I started writing this while preparing for interviews, and it was intended as a personal reference that I could easily navigate using VSCode navigation features.

  But I provide it here in case anyone else finds this useful.

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

  Directory Structure

  minllm/
  ├── papers/              # Research paper summaries and notes
  │   ├── data/           # Data processing and corpus papers
  │   ├── posttraining/   # Instruction and preference tuning papers
  │   └── pretraining/    # Core transformer architecture papers
  ├── pretraining/        # Core pretraining implementation
  │   ├── common/         # Shared PyTorch modules and patterns
  │   │   ├── base/      # Base classes for LLM components
  │   │   └── patterns/  # Reusable architectural patterns
  │   ├── configs/       # Model and training configurations
  │   ├── data/          # Data loading and processing utilities
  │   ├── trainer/       # Training loop implementation
  │   └── utils/         # Training utilities (distributed, checkpointing, etc.)
  ├── posttraining/       # Fine-tuning implementations
  │   └── instruction_tuning/  # SFT (Supervised Fine-Tuning) implementation

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

I reorganized a lot of their code, annotated stuff, and invariably made it worse to use -- but easier (for me) to read.

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
