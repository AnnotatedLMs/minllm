## TODO FIX -- ASK FOR PROGRESSION
## TODO ADD SURVEY PAPER

# Comprehensive Position Encoding Papers Collection

## Foundational Paper

### **Attention Is All You Need (2017)**
- Vaswani et al.
- Introduced **Sinusoidal Position Embeddings**
- Historical importance: Used in original Transformer, BERT, early GPT models
- https://arxiv.org/pdf/1706.03762

## Position Encoding Methods in Major Contemporary LLMs

### Rotary Position Embeddings (RoPE)

#### **RoFormer/RoPE (2021)**
- Su et al.
- **Rotary Position Embedding** - rotates query/key vectors based on position
- **Used in**: **LLaMA 1/2/3**, **Mistral/Mixtral**, **Qwen**, **DeepSeek**, **Phi**, **Yi**, **Gemma**
- The dominant position encoding method in modern LLMs
- https://arxiv.org/pdf/2104.09864

### Linear Attention Biases

#### **ALiBi (2022)**
- Press et al.
- **Attention with Linear Biases** - simple linear penalty based on distance
- **Used in**: **BLOOM**, **MPT models**
- https://arxiv.org/pdf/2108.12409

### Context Extension Methods

#### **Position Interpolation (2023)**
- Chen et al.
- Extends context window of RoPE models via interpolation
- **Used in**: **Code Llama**, **LLaMA 2 Long**, many extended-context variants
- https://arxiv.org/pdf/2306.15595

#### **YaRN (2023)**
- Peng et al.
- **Yet another RoPE extensioN** - improves position interpolation
- **Used in**: Extended context versions of open models (Nous-Hermes, etc.)
- https://arxiv.org/pdf/2309.00071

## Position Encoding by Major Model Family

### Models Using RoPE
- **Meta**: LLaMA 1/2/3, Code Llama
- **Mistral AI**: Mistral, Mixtral
- **Alibaba**: Qwen series
- **DeepSeek**: All DeepSeek models
- **Microsoft**: Phi series
- **01.AI**: Yi models
- **Google**: Gemma

### Models Using ALiBi
- **BigScience**: BLOOM
- **MosaicML**: MPT series

### Models Using Custom/Minimal Position Encoding
- **OpenAI GPT-3.5/4**: Proprietary (likely modified from GPT-3's learned embeddings)
- **Anthropic Claude**: Proprietary
- **Google Gemini**: Proprietary (reportedly minimal explicit position encoding)

## Key Surveys

### **Length Extrapolation of Transformers: A Survey from the Perspective of Positional Encoding (2024)**
- Zhao et al.
- Comprehensive survey of position encoding methods for length extrapolation
- https://arxiv.org/pdf/2312.17044v4

## Critical Insights

### Why RoPE Dominates
- Better performance than sinusoidal embeddings
- More efficient than learned absolute position embeddings
- Naturally extends to longer sequences than trained on
- Can be further extended via interpolation methods

### The Context Extension Revolution
- Position Interpolation and its variants enabled extending pre-trained models from 2K-4K tokens to 128K+ tokens
- Critical for modern long-context applications
- Most 100K+ context models use RoPE + some form of interpolation
