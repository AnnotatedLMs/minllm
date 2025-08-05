  # MinLLM - An Opinionated Guide for Transformer-based LLMs

  ## Overview

  This repository is a reference guide for transformer-based Large Language Models (LLMs). The code is intentionally approximate, not optimized, nor well engineered.

  The organization is very overly granular -- it's hard for me to learn stuff when everything is smushed together.

  I started writing this while preparing for interviews, and it was primarily intended as a personal reference that I could easily navigate using VSCode navigation features.

  But I provide it here in case anyone wants to peek at stuff.

  ## Installation Guide

  1. **Install uv (Python package manager)**
     ```bash
     # Follow instructions at https://github.com/astral-sh/uv
     curl -LsSf https://astral.sh/uv/install.sh | sh

  2. Set up virtual environment
  uv venv
  3. Activate virtual environment
  # On macOS/Linux:
  source .venv/bin/activate
  4. Install dependencies
  uv sync
  5. Install pre-commit hooks
  uv run pre-commit install

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

| Model | Status |
|-------|--------|
| GPT-2 | ✅ |
| LLaMA 3.1 | ✅ |
| DeepSeek 3 | ✅ |
| Qwen 3 | ❌ |
| Kimi k2 | ❌ |

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
