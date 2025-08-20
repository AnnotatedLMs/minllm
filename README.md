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

# Install all dependencies
uv sync

uv run pre-commit install
```

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

I reorganized a lot of their code, annotated stuff, and invariably made it worse to use -- but easier (for me) to read.
