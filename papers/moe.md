## TODO FIX -- ASK FOR PROGRESSION
# Comprehensive Mixture of Experts (MoE) Papers Collection

## Foundational Papers (1990s)
- **Adaptive Mixtures of Local Experts (1991)** - Jacobs et al.
  - The original MoE paper that introduced the concept
  - https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf

- **Hierarchical Mixtures of Experts and the EM Algorithm (1994)** - Jordan & Jacobs
  - Extended MoE to hierarchical structures
  - https://www.cs.toronto.edu/~hinton/absps/hme.pdf

## Modern Era MoE Papers

### Breakthrough Paper (2017)
- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017)** - Shazeer et al.
  - Brought MoE to modern deep learning and NLP with up to 137B parameters
  - https://arxiv.org/pdf/1701.06538

### Major MoE Models and Architectures

#### Infrastructure/Research Models
- **GShard (2020)** - Google
  - Scaled MoE to 600B parameters
  - https://arxiv.org/pdf/2006.16668

- **Switch Transformers (2021)** - Google
  - Simplified MoE with single expert routing, scaled to 1.6T parameters
  - https://arxiv.org/pdf/2101.03961

- **GLaM (2021)** - Google
  - 1.2T parameter model, more compute-efficient than dense models
  - https://arxiv.org/pdf/2112.06905

- **ST-MoE (2022)** - Google
  - Focused on training stability and transfer learning
  - https://arxiv.org/pdf/2202.08906

#### Production/Open Models
- **Mixtral-8x7B (2023)** - Mistral AI
  - First widely-used open MoE model
  - https://arxiv.org/pdf/2401.04088

- **DeepSeekMoE (2024)** - DeepSeek
  - Fine-grained expert segmentation
  - https://arxiv.org/pdf/2401.06066

- **DeepSeek-V2 (2024)** - DeepSeek
  - Multi-head Latent Attention with MoE
  - https://arxiv.org/pdf/2405.04434

- **DeepSeek-V3 (2024)** - DeepSeek
  - Advanced MoE with auxiliary loss improvements
  - https://arxiv.org/pdf/2412.19437

- **DeepSeek-R1 (2025)** - DeepSeek
  - Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
  - https://arxiv.org/pdf/2501.12948

- **OLMoE (2024)** - Allen AI
  - Open Mixture-of-Experts Language Models
  - https://arxiv.org/pdf/2409.02060

- **Kimi 1.5 (2025)** - Moonshot AI
  - https://arxiv.org/pdf/2501.12599

### Multimodal MoE Models
- **Gemini 1.5 (2024)** - Google
  - Multimodal understanding across millions of tokens
  - https://arxiv.org/pdf/2403.05530

- **Gemini 2.5 (2024)** - Google
  - https://arxiv.org/pdf/2507.06261

- **MoE-LLaVA (2024)**
  - Vision-language MoE model
  - https://arxiv.org/pdf/2401.15947

- **V-MoE (2021)** - Google
  - Vision Transformer with MoE
  - https://arxiv.org/pdf/2106.05974

### Key Algorithmic Papers
- **Mixture-of-Experts with Expert Choice Routing (2022)** - Google
  - Used in Gemini models - inverts token-choice to expert-choice
  - https://arxiv.org/pdf/2202.09368

- **Sparse Upcycling (2022)** - Google
  - Used to initialize MoE from dense checkpoints (used in practice)
  - https://arxiv.org/pdf/2212.05055

### Surveys
- **A Review of Sparse Expert Models in Deep Learning (2022)** - Fedus et al.
  - https://arxiv.org/pdf/2209.01667

- **A Survey on Mixture of Experts in Large Language Models (2024)** - Cai et al.
  - Most comprehensive recent survey
  - https://arxiv.org/pdf/2407.06204
