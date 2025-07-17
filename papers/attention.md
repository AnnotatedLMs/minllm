## TODO FIX -- ADD LATENT ATTENTION, ASK FOR PROGRESSION
## ADD https://arxiv.org/pdf/2505.13544 Multi-head Temporal Latent Attention (2025) Deng, Woodland

# Essential Attention Papers for Modern LLMs

## Foundational Papers

### The Beginning
- **Attention Is All You Need (2017)** - Vaswani et al.
  - Introduced self-attention and the Transformer architecture
  - Core mechanism in ALL modern LLMs (GPT, BERT, LLaMA, etc.)
  - https://arxiv.org/pdf/1706.03762

## Efficiency Improvements in Production

### Memory-Efficient Attention
- **Multi-Query Attention (2019)** - Shazeer
  - Reduces KV cache memory usage dramatically
  - Used in: Original PaLM
  - https://arxiv.org/pdf/1911.02150

- **Grouped-Query Attention (2023)** - Ainslie et al.
  - Sweet spot between MHA and MQA
  - Used in: LLaMA-2, LLaMA-3, Mistral, Gemma, most 2023+ models
  - https://arxiv.org/pdf/2305.13245

### Speed Optimizations
- **FlashAttention (2022)** - Dao et al.
  - 2-3x training speedup through IO-aware implementation
  - Used in: Training infrastructure for most modern LLMs
  - https://arxiv.org/pdf/2205.14135

- **FlashAttention-2 (2023)** - Dao et al.
  - Further 2x improvement
  - Standard in current LLM training stacks
  - https://arxiv.org/pdf/2307.08691

## Long Context Handling

### Sparse Patterns
- **Sparse Transformers (2019)** - OpenAI
  - Factorized sparse attention patterns
  - Used in: GPT-3's sparse attention layers
  - https://arxiv.org/pdf/1904.10509


## Recent Production Advances

### Attention Variants in Latest Models
- **Multi-Head Latent Attention (MLA)** - DeepSeek-V2 (2024)
  - Low-rank KV compression for efficiency
  - Used in: DeepSeek-V2, DeepSeek-V3
  - https://arxiv.org/pdf/2405.04434

- **Differential Transformer (2024)** - Microsoft
  - Noise reduction in attention mechanism
  - Used in: DIFF Transformer models
  - https://arxiv.org/pdf/2410.05258

## Surveys

### Attention Mechanism Analysis
- **Attention Heads of Large Language Models: A Survey (2024)** - Zheng et al.
  - Comprehensive survey of attention head functions and mechanisms
  - Analyzes how different heads specialize (e.g., syntax, semantics, reasoning)
  - https://arxiv.org/pdf/2409.03752

## Key Concepts Summary

**Every modern LLM uses:**
- Multi-head self-attention (from "Attention Is All You Need")
- Either RoPE or ALiBi for positions
- GQA or MQA for inference efficiency
- FlashAttention for training efficiency

**For long context:**
- Sliding windows (Mistral)
- Sparse patterns (GPT-3)
- Extended/recurrent methods
