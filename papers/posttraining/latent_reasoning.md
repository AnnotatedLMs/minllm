# Latent Reasoning in Large Language Models

## Overview
Latent reasoning represents an alternative approach to how LLMs perform complex reasoning tasks, shifting from explicit chain-of-thought (CoT) generation to computation within continuous hidden states. This eliminates token-level supervision while maintaining multi-step inference capabilities.

## Development Timeline

### Phase 1: Chain-of-Thought Foundation (2022)
**Core concept**: LLMs generate intermediate reasoning steps in natural language before final answers, demonstrating that models benefit from additional computation time for problem-solving.

**Key characteristics**:
- Explicit verbalization of intermediate steps
- Significant performance gains on reasoning tasks
- High inference latency from token generation
- Human-interpretable reasoning traces

**Design patterns**: Sequential generation of reasoning tokens, typically using prompts like "Let's think step by step."

**Technical insights**: This approach revealed the expressive bandwidth limitation of natural language - approximately 15 bits per token versus 40,960 bits per hidden state.

**Chain-of-thought prompting elicits reasoning in large language models** (2022) - Wei et al. (Google)
- https://arxiv.org/abs/2201.11903

### Phase 2: Architectural Approaches - Activation-Based Recurrence (2018-2024)

**Core concept**: Activation-based methods create recurrent computational flows by reprocessing information through the same layers multiple times, expanding computational depth without generating tokens.

**Key characteristics**:
- Shared parameters across iterations (vertical recurrence)
- Adaptive or fixed computation depth
- No explicit reasoning token generation
- Emergence of Pre/Loop/Coda architectural patterns

The field split into two contrasting approaches to implementing recurrence:

#### Dynamic Depth with Adaptive Computation
These approaches let the model decide how much computation each token needs, using learnable mechanisms to allocate depth dynamically - more complex problems get more iterations automatically.

**Universal Transformers** (2018) - Dehghani et al.
- https://arxiv.org/pdf/1807.03819
- Adaptive Computation Time (ACT) mechanism
- Dynamic recurrence over layers with per-position halting

**CoTFormer** (2023) - Mohtashami et al.
- https://arxiv.org/pdf/2310.10845
- Mixture of Reasoning (MoR) router for adaptive depth
- Different tokens receive different computation budgets

#### Fixed-Depth Structured Recurrence
In contrast to adaptive approaches, these methods use predetermined iteration counts with explicit architectural stages, sacrificing flexibility for simplicity and training stability.

**Recursive Transformer** (2024) - Bae et al.
- https://arxiv.org/pdf/2410.20672
- Share/refill mechanism for hidden state management
- Layer-wise LoRA for parameter efficiency

**AlgoFormer** (2024) - Gao et al.
- https://arxiv.org/pdf/2402.13572
- Explicit Pre/Loop/Coda structure
- Fixed iterations without complex halting mechanisms

**Design patterns**:
- Evolution from complex adaptive mechanisms to simpler fixed patterns
- Shift from monolithic to modular architectures
- Deprecation of depth embeddings in fixed-depth models

**Technical insights**: The field has moved toward simpler, more stable designs, suggesting that fixed-depth recurrence with proper structure can match adaptive approaches while being easier to train and deploy.

### Phase 3: Training-Induced Latent Reasoning (2024-2025)

**Core concept**: Training-induced methods create implicit loops in the computation graph through continuous activation recurrence, compressed state representations, or strategic token insertion - all without modifying the standard Transformer architecture.

**Key characteristics**:
- No architectural modifications required
- Works with pretrained models
- Various induction strategies (continuous, compressed, token-based)
- Curriculum learning often necessary

Three distinct strategies emerged for inducing recurrence through training:

#### Continuous Activation Recurrence
These methods create explicit loops by feeding the model's full hidden states back as inputs, maintaining complete information flow but requiring careful training to prevent instability.

**Coconut** (2024) - Hao et al. (Meta)
- https://arxiv.org/pdf/2412.06769
- Loops LLM's last hidden state back as input
- Enables breadth-first exploration in latent space
- Curriculum learning for stability

**CODI** (2025) - Shen et al.
- https://arxiv.org/pdf/2502.21074
- Self-distillation to align teacher (with CoT) and student (without)
- Single-step alignment more stable than curriculum
- First to match explicit CoT on GSM8K

#### Compressed State Recurrence
Rather than looping full activations, these methods compress reasoning into discrete checkpoints, trading some information loss for more efficient recurrence and easier training dynamics.

**Light Thinker** (2025) - Zhang et al.
- https://arxiv.org/pdf/2502.15589
- "Gist tokens" as compression anchors
- Transforms horizontal reasoning into vertical computation

**VO-VAE** (2025) - Su et al.
- https://arxiv.org/pdf/2502.03275
- Replaces early CoT segments with discrete latent tokens
- Hierarchical recurrence through abstract tokens

#### Iteration Expansion through Strategic Tokens
Instead of explicit loops or compression, these methods simply insert special tokens that give the model more processing steps, exploiting the fact that additional attention operations enable deeper reasoning.

**Pause Tokens** (2024) - Goyal et al.
- https://arxiv.org/pdf/2310.02226
- Learnable tokens explicitly signal computation steps
- Simple but effective approach

**Planning Tokens** (2024) - Wang et al.
- https://arxiv.org/pdf/2310.05707
- Hierarchical recurrence structure
- Each token initiates new reasoning loop

**Filler Tokens** (2024) - Pfau et al.
- https://arxiv.org/pdf/2404.15758
- Even meaningless tokens improve reasoning
- Demonstrates reasoning emerges from additional attention steps

**Design patterns**:
- Convergence toward simpler implementations
- Mixture of continuous and discrete approaches
- Token insertion as lightweight recurrence induction

**Technical insights**: These methods reveal that recurrence for reasoning is not solely architectural but can emerge from appropriate training objectives. Success across different implementations suggests the specific method matters less than ensuring sufficient computational depth.

### Phase 4: Hidden State-Based Methods (2024-2025)

**Core concept**: Hidden state methods convert Transformers to use compressed fixed-size memory instead of growing KV caches, enabling recurrent-like processing while maintaining efficiency.

**Key characteristics**:
- Fixed-size memory regardless of sequence length
- Focus on Transformer-to-RNN conversions
- Training through distillation or fine-tuning
- Addresses KV cache memory bottleneck

Two approaches emerged for managing hidden states:

#### Gradient-State Recurrence (Test-Time Training Family)
These methods treat hidden states as learnable parameters that are optimized during inference itself, effectively training a small part of the model on each new input sequence.

**TTT** (2024) - Sun et al.
- https://arxiv.org/pdf/2407.04620
- Hidden states updated via SGD during inference
- Each token performs one optimization step
- Demonstrates depth-time equivalence

**Titans** (2025) - Behrouz et al.
- https://arxiv.org/pdf/2501.00663
- Incorporates Adam-like momentum
- 250M model matches 1.3B Transformer performance

**ATLAS** (2025) - Behrouz et al.
- https://arxiv.org/pdf/2505.23735v1
- Second-order optimization (Muon)
- Maintains auxiliary optimization state

#### Training-Induced Hidden-State Conversion
Unlike gradient-state methods that optimize during inference, these approaches convert entire pretrained Transformers into recurrent architectures through one-time distillation or fine-tuning procedures.

**SUPRA** (2024) - Mercat et al.
- https://arxiv.org/pdf/2405.06640
- Converts Llama/Mistral to linear attention
- Only 5% of original training cost

**MOHAWK** (2024) - Bick et al.
- Transformer to Mamba-2 conversion
- Three-phase distillation procedure

**Llamba** (2025) - Bick et al.
- https://arxiv.org/abs/2502.14458
- Scales to 1-8B models
- Matches Llama-3 with 0.1% training compute

**LoLCATs** (2024) - Zhang et al.
- https://arxiv.org/pdf/2410.10254
- Low-rank linearization approach
- Only 0.2% parameter updates needed

## Training Pipeline Integration

### Typical Integration Points

1. **During Continued Pretraining**
   - Architectural methods requiring specialized architectures
   - Example: Universal Transformer variants

2. **During Instruction Tuning** ← **Most Common**
   - Training-induced methods (Coconut, pause tokens)
   - Combined with standard SFT objectives
   - 10-100B token requirements typical

3. **After Preference Tuning**
   - Compression of explicit reasoning
   - Examples: Stepwise Internalization, CODI

4. **At Inference Time**
   - TTT-family methods
   - No training required

## Practical Considerations

### Implementation Challenges

**Memory and Computation**
- Hidden state management in recurrent architectures
- Gradient storage for optimization-based methods
- Chunk-wise parallelization complexity
- GPU memory constraints on depth

**Training Stability**
- Gradient issues in iterative architectures
- Curriculum learning complexity
- Convergence slower than standard training
- Hyperparameter sensitivity

**Inference Trade-offs**
- Latency vs. reasoning quality balance
- Unpredictable adaptive computation time
- Parallelization requirements
- Real-time application constraints

### Common Pitfalls

**Training Issues**
- Over-aggressive reasoning compression
- Insufficient curriculum design
- Architecture-method mismatches
- Poor scaling to larger models

**Evaluation Gaps**
- Testing only in-distribution
- Missing multi-step reasoning validation
- Over-reliance on final accuracy metrics
- Failure to detect reasoning shortcuts

## Historical Gist

### Brief Timeline

**2018**: Universal Transformer introduces adaptive computation time for variable-depth processing

**2022**: Chain-of-thought prompting shows explicit reasoning steps improve performance

**2023**: CoTFormer demonstrates architectural recurrence without token generation

**2024**: Coconut and pause tokens show standard Transformers can learn latent reasoning through training alone

**2024-2025**: Multiple approaches emerge simultaneously:
- Simplified architectural methods (AlgoFormer, Recursive Transformer)
- Various training-induced methods (CODI, Light Thinker, planning tokens)
- Hidden state conversions (SUPRA, Llamba)
- Test-time optimization methods (TTT, Titans, ATLAS)

### Some Trends

- Early architectural approaches used complex adaptive mechanisms; later ones prefer fixed depth
- Shift from modifying architectures to modifying training procedures
- Multiple independent methods achieve similar performance gains
- Memory constraints (KV cache) drive development of fixed-size state methods

### Some Themes

- Reasoning benefits from iterative refinement, regardless of implementation
- Trade-offs between interpretability (explicit CoT) and efficiency (latent methods)
- Sufficient computational depth matters more than specific approach
- Standard Transformers have latent capacity for recurrent-like computation

## Surveys

**A Survey on Latent Reasoning** (2025) - Zhu et al.
- https://arxiv.org/pdf/2507.06203
