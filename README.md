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

# Install dependencies based on your hardware:

# Choose one:
# For CPU-only development/testing:
uv sync --extra cpu
# For GPU training (CUDA 12.8):
uv sync --extra cu128

# Install pre-commit hooks
uv run pre-commit install
```

### Installation Options

#### CPU Installation
For development or testing without GPU acceleration:
```bash
uv sync --extra cpu
```
- PyTorch CPU-only builds
- All base dependencies

#### GPU Installation (CUDA 12.8)
For training with GPU acceleration:
```bash
uv sync --extra cu128
```
- PyTorch with CUDA 12.8 support
- All base dependencies

##### Optional: FlashMLA Installation (Linux + NVIDIA GPU only)
For optimized DeepSeek3 attention on compatible GPUs (SM90/SM100):
```bash
# After running uv sync --extra cu128
git clone https://github.com/deepseek-ai/FlashMLA.git
cd FlashMLA
uv pip install -v .
cd ..
```
**Note:** FlashMLA requires NVIDIA GPU with SM90/SM100 architecture (Ada Lovelace/Hopper), CUDA 12.x, and Linux. It will be automatically detected and used if available.

### Optional Setup

```bash
# Install only production dependencies (no dev tools)
uv sync --no-dev --extra cpu  # or --extra cu128
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

When you run `uv sync --extra gpu`, here's how the configuration works:

### 1. Base Dependencies (always installed)
```toml
[project]
dependencies = [
    "torch>=2.7.1",          # Always installed (source depends on extra)
    "accelerate>=1.8.1",
    "datasets>=3.6.0",
    # ... etc
]
```

### 2. Optional Dependencies (activated by --extra flag)
```toml
[project.optional-dependencies]
cu128 = [
    "flash-mla",         # Only installed with --extra cu128
]
```

### 3. Source Resolution (WHERE to get packages)
```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", extra = "cpu" },     # Used with --extra cpu
  { index = "pytorch-cu128", extra = "cu128" }, # Used with --extra cu128
]
```
**Note:** The `extra` markers mean you MUST specify either `--extra cpu` or `--extra cu128` when running `uv sync`, otherwise torch will be installed from PyPI (default).

### 4. Index Definitions (what index names mean)
```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
```

### 5. Conflicts (prevent mixing incompatible extras)
```toml
[tool.uv]
conflicts = [
  [{ extra = "cpu" }, { extra = "cu128" }]  # Can't use both together
]
```

### Full Flow
- `uv sync --extra cpu`: Installs torch from CPU index
- `uv sync --extra cu128`: Installs torch from CUDA 12.8 index
- `uv sync` alone: Would install torch from PyPI (not recommended)
- FlashMLA: Must be manually installed for GPU users (see Installation Options)
