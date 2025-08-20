# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn


class MLACompressionMixin:
    """
    Mixin for Multi-head Latent Attention (MLA) compression.
    https://arxiv.org/pdf/2412.19437

    Significance:
    Makes attention use less memory by squishing inputs into smaller spaces first.
    Instead of working with huge hidden_dim, works with smaller compression_dim.

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
    - Q compression: Learns which dimensions of the input create high dot products with relevant keys
    - KV compression: Learns which dimensions help tokens match queries (K) and which contain useful content (V)
    - Both throw away redundant dimensions that don't affect attention patterns

    Used by: DeepSeek-V3's Multi-head Latent Attention
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
