# Preference Tuning

## Overview
Preference tuning encompasses techniques for teaching LLMs to better align with human preferences and expectations.

## The Evolution of Preference Tuning

### Phase 1: Foundation with RLHF (2022)
**InstructGPT/ChatGPT** pioneered the modern RLHF pipeline:
- **Training language models to follow instructions with human feedback** (2022) - Ouyang et al. (OpenAI)
  - https://arxiv.org/pdf/2203.02155

**The three-stage pipeline:**
1. **Supervised Fine-Tuning (SFT)**: Train on high-quality instruction-following demonstrations
2. **Reward Model Training**: Train a separate model to predict human preferences from ranked outputs
3. **PPO Optimization**: Use the reward model's scores to update the LLM via Proximal Policy Optimization

**Key clarification**: A KL divergence constraint prevents the model from exploiting the reward model.

### Phase 2: RLAIF - Reinforcement Learning from AI Feedback (2022-2023)

**Conceptual shift**: RLAIF maintains the same three-stage pipeline as RLHF but replaces human annotators with AI systems for generating preference labels, dramatically reducing cost and scaling bottlenecks.

**Key papers**:
- **Constitutional AI: Harmlessness from AI Feedback** (2022) - Bai et al. (Anthropic)
  - https://arxiv.org/pdf/2212.08073
- **RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback** (2023) - Lee et al. (Google)
  - https://arxiv.org/pdf/2309.00267

#### Constitutional AI Pipeline (Anthropic's approach)
1. **SFT on helpful data**: Initial supervised fine-tuning
2. **AI-generated critiques and revisions**: Model critiques its own outputs based on constitutional principles and revises them
3. **Train on preference pairs**: Use (original, revision) pairs as preference data with standard RLHF techniques

#### Standard RLAIF Pipeline (Google's generalization)
1. **Start with SFT model**: Same as RLHF
2. **Generate preference labels with LLM**:
   - Use an off-the-shelf LLM to rate pairs of responses
   - Mitigate position bias by averaging preferences with swapped order
   - Optional: Use chain-of-thought prompting for better alignment
3. **Train reward model**: Distill AI preferences into a reward model
4. **RL fine-tuning**: Standard PPO optimization using the AI-trained reward model

#### Key RLAIF Variants (from Lee et al., 2023)
- **Direct-RLAIF (d-RLAIF)**: Addresses reward model "staleness" (a known problem in both RLHF and standard RLAIF)
  - Problem: Reward models become outdated as policies improve during training
  - Solution: Skip reward model entirely; LLM directly scores responses during RL (1-10 scale)
  - Result: Superior performance to canonical RLAIF

- **Same-size RLAIF**: Google's experiment testing self-improvement limits
  - Question: Does RLAIF require a larger/better model as the labeler?
  - Setup: AI labeler is same size as policy being trained
  - Finding: Still achieves significant gains over SFT baseline
  - Implication: Models can potentially improve themselves

**Performance**: Researchers argue that RLAIF achieves comparable performance to RLHF across summarization, helpful dialogue, and harmless dialogue tasks (Lee et al., 2023). The approach is approximately 10x more cost-effective than human annotation.

### Phase 3: Simplification and Efficiency Innovations (2023-2024)

**Conceptual shift**: Researchers explore different ways to make preference tuning more efficient - either by eliminating components entirely (DPO) or by making RL more memory-efficient (GRPO).

#### Direct Preference Optimization (DPO)
**Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (2023) - Rafailov et al.
- https://arxiv.org/pdf/2305.18290

**Key innovation**: DPO eliminates the need for reward modeling and RL entirely, showing that preference tuning can be reduced to a single supervised learning objective.

**Pipeline for DPO:**
1. **Start with SFT model**: Same as RLHF/RLAIF
2. **Collect preference pairs**: Same as RLHF (human) or RLAIF (AI)
3. **Direct optimization**: Train directly on preference pairs without reward model or PPO

#### Group Relative Policy Optimization (GRPO)
**DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (2024) - Shao et al.
- https://arxiv.org/pdf/2402.03300

**Key innovation**: GRPO maintains the RL approach but eliminates the value model (critic) that PPO requires, making it significantly more memory-efficient while maintaining effectiveness.

**Pipeline for GRPO:**
1. **Start with SFT model**: Same as RLHF/RLAIF
2. **Collect preference data**: Can use human or AI preferences
3. **Group-based RL optimization**:
   - Sample multiple outputs (e.g., 64) for each prompt
   - Use average reward of the group as baseline (instead of learned value function)
   - Update policy using relative advantages within each group

**Advantages over PPO**:
- **Memory efficiency**: No separate value model to train/store
- **Stability**: Group normalization provides natural variance reduction
- **Simplicity**: Fewer hyperparameters to tune

**GRPO Variants**:
- **Outcome supervision**: Reward only at end of response
- **Process supervision**: Rewards at each reasoning step (better for complex tasks)
- **Iterative GRPO**: Continuously update reward model during training

## Key Surveys and Resources

1. **Preference Tuning with Human Feedback on Language, Speech, and Vision Tasks: A Survey** (2024)
   - Wang et al., Columbia University
   - Comprehensive coverage including multimodal preference tuning
   - https://www.columbia.edu/~wt2319/Preference_survey.pdf

2. **Aligning Large Language Models with Human Preferences** (2023)
   - Wang et al., Huawei Noah's Ark Lab
   - Focus on recent advances in alignment techniques
   - https://arxiv.org/pdf/2307.12966

## Documented Implementations

### Models Using RLHF/PPO
- **ChatGPT/GPT-4** (OpenAI) - Original implementation
- **Claude** (Anthropic) - Constitutional AI + RLHF
- **Llama 2-Chat** (Meta) - PPO with rejection sampling
- **Gemini** (Google) - PPO-based RLHF

### Models Using DPO
- **Mistral-7B-Instruct** - DPO for instruction tuning
- **Zephyr** - Built on Mistral using DPO
- **Intel Neural-Chat** - Leverages DPO for alignment
- **Tulu 2** (Allen AI) - Iterative DPO refinement

### Models Using GRPO
- **DeepSeekMath** - First implementation, achieved state-of-the-art math performance
- **DeepSeek-Coder** - Applied GRPO to code generation tasks

## Key Technical Insights

### Why DPO
- Showed that RLHF's reward modeling and RL optimization could be combined into a single supervised learning objective
- Eliminated training instabilities associated with PPO
- Reduced computational requirements significantly

### Why GRPO
- Maintains the benefits of RL-based optimization (can handle complex reward signals)
- Reduces memory requirements by ~50% compared to PPO
- Particularly effective for domains with clear evaluation metrics (math, code)
- Can incorporate fine-grained process supervision for multi-step reasoning

## Practical Considerations

### Common Challenges
- PPO requires careful hyperparameter tuning and can be unstable
- DPO is generally more stable but still requires careful batch size selection
- GRPO offers a middle ground - more stable than PPO but retains RL flexibility
- All methods need KL regularization to prevent mode collapse
