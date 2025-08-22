# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn


class MLACompressionMixin:
    """
    Mixin for Multi-head Latent Attention (MLA) compression.

    Scholarship:
    Deepseek-V2, 2024, https://arxiv.org/pdf/2405.04434, 2.1.2
    Deepseek-V3, 2025, https://arxiv.org/pdf/2412.19437, 2.1.1

    Significance:
    Makes attention use less memory by squishing inputs into smaller spaces first.
    Learned linear weights ensure we compress in a way that doesn't hurt downstream performance.

    Init:
    The compression layers are defined in MultiHeadLatentAttention as:
        self.kv_down = nn.Linear(hidden_dim, kv_compression_dim)
        self.query_down = nn.Linear(hidden_dim, query_compression_dim)

    Step-by-step control flow:
    1. Take input (big dimension)
    2. Squish it down to KV compression size
    3. Separately squish it down to Q compression size
    4. Return both small versions

    Learning process:
    - Q compression:
      - Learns to transform inputs into query vectors that have high similarity with relevant keys
      - When predicted token is wrong: loss increases, producing larger gradients
      - Gradients signal that current query vectors have low similarity with keys of helpful tokens
      - Query projection weights adjust to transform inputs into queries that better align with useful keys
      - Result: the linear transformation learns to produce compressed queries that attend to relevant context

    - KV compression:
      - K (Key): Learns to transform inputs into keys that are distinguishable by queries
        - When wrong tokens get high attention: gradients signal keys aren't differentiating helpful from unhelpful tokens
        - Key projection weights adjust to produce keys that make relevant tokens more distinct
        - Result: the transformation learns to create keys that help queries find the right tokens

      - V (Value): Learns to transform inputs into values containing prediction-relevant information
        - When prediction is wrong: gradients signal that attended values lack useful information
        - Value projection weights adjust to preserve information dimensions that improve predictions
        - Result: the transformation learns to extract and compress the most predictive features

    - The weight matrices learn transformations that preserve task-relevant information while compressing dimension
    """

    def _compress_inputs(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        kv_down_proj: nn.Linear,
        q_down_proj: nn.Linear,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq_len kv_compress_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq_len q_compress_dim"],
    ]:
        """Compress inputs to latent dimensions."""
        kv_compressed: jaxtyping.Float[torch.Tensor, "batch seq_len kv_compress_dim"]
        kv_compressed = kv_down_proj(x)

        q_compressed: jaxtyping.Float[torch.Tensor, "batch seq_len q_compress_dim"]
        q_compressed = q_down_proj(x)

        return kv_compressed, q_compressed
