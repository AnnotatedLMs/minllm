# Standard Library
import math
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models.attention import attention_mixins
from pretraining.common.models.attention import compression_mixins
from pretraining.common.models.attention import flash_mla_mixin
from pretraining.common.models.attention import projection_mixins
from pretraining.common.models.position import partial_rope
from pretraining.common.models.position import position_mixins


class MultiHeadLatentAttention(
    nn.Module,
    compression_mixins.MLACompressionMixin,
    projection_mixins.MLAProjectionMixin,
    position_mixins.PartialRoPEApplicationMixin,
    attention_mixins.ManualAttentionMixin,
    attention_mixins.MultiHeadReshapeMixin,
    flash_mla_mixin.FlashMLAMixin,
):
    """
    Multi-head Latent Attention (MLA) - Compression-based attention for large models.
    https://arxiv.org/pdf/2405.04434 Section 2.1
    https://arxiv.org/pdf/2412.19437 Section 2.1.1

    Inspired by:
    MLA-MoE, Deepseek-V2 - https://arxiv.org/abs/2405.04434

    Step-by-step control flow (how mixins work together):
    1. MLACompressionMixin: Compress input to small latent spaces (KV and Q separately)
    2. MLAProjectionMixin: Expand KV latent to keys_content, values; keys_rope from input
    3. MLAProjectionMixin: Expand Q latent to queries_content, queries_rope
    4. MultiHeadReshapeMixin: Reshape flat projections to multi-head format
    5. PartialRoPEApplicationMixin: Apply RoPE to position features, concat with content
    6. MultiHeadReshapeMixin: Transpose to attention format [batch, heads, seq, dim]
    7. ManualAttentionMixin: Compute attention scores, apply softmax, weighted sum
    8. MultiHeadReshapeMixin: Merge heads back to flat representation
    9. Output projection: Final linear layer back to hidden_dim

    Learning process (how each mixin affects training):
    - MLACompressionMixin: Down-projections learn to extract essential features
    - MLAProjectionMixin: Up-projections learn to reconstruct attention components
    - PartialRoPEApplicationMixin: NO learning - just applies sinusoidal position encoding
    - ManualAttentionMixin: NO learning - just computes attention mechanism
    - Output projection: Learns to combine multi-head outputs

    Key implementation detail:
    - Keys RoPE: Projects directly from input (W_KR * h_t) per equation 3
    - Queries RoPE: Projects from compressed query (W_QR * c_Q) per equation 8
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        kv_compression_dim: int,
        query_compression_dim: int,
        key_rope_module: partial_rope.PartialRoPE,  # Separate module for keys (can have YaRN)
        query_rope_module: partial_rope.PartialRoPE,  # Separate module for queries (no YaRN)
        rope_dim: int = 64,
        dropout: float = 0.0,
        is_causal: bool = True,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.key_rope_module = key_rope_module
        self.query_rope_module = query_rope_module
        self.is_causal = is_causal
        self.use_flash_attention = use_flash_attention

        # Compression layers
        self.kv_down = nn.Linear(hidden_dim, kv_compression_dim)
        self.query_down = nn.Linear(hidden_dim, query_compression_dim)

        # Separate up-projections for keys, values, and queries (from compressed)
        self.key_up = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.value_up = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.query_up = nn.Linear(query_compression_dim, num_heads * head_dim)

        # RoPE projections (key from input, query from compressed per paper)
        self.key_rope = nn.Linear(hidden_dim, num_heads * rope_dim)  # W_KR in paper
        self.query_rope = nn.Linear(query_compression_dim, num_heads * rope_dim)  # W_QR in paper

        # Output projection
        self.output_proj = nn.Linear(num_heads * head_dim, hidden_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout if dropout is not None else 0.0)

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Apply Multi-head Latent Attention."""
        batch_size, seq_len, _ = x.shape

        # Step 1: Compress inputs
        kv_compressed, query_compressed = self._compress_inputs(x, self.kv_down, self.query_down)

        # Step 2: Project KV to content and position features
        keys_content, values, keys_rope = self._project_compressed_kv(
            x,
            kv_compressed,
            self.key_up,
            self.value_up,
            self.key_rope,
            self.num_heads,
            self.head_dim,
            self.rope_dim,
        )

        # Step 3: Project queries to content and position features
        queries_content, queries_rope = self._project_compressed_queries(
            query_compressed,
            self.query_up,
            self.query_rope,
            self.num_heads,
            self.head_dim,
            self.rope_dim,
        )

        # Step 4: Reshape to multi-head format
        # Keys content: [batch, seq, num_heads * head_dim] -> [batch, seq, heads, head_dim]
        keys_content = keys_content.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries_content = queries_content.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # RoPE features: [batch, seq, num_heads * rope_dim] -> [batch, seq, heads, rope_dim]
        keys_rope = keys_rope.view(batch_size, seq_len, self.num_heads, self.rope_dim)
        queries_rope = queries_rope.view(batch_size, seq_len, self.num_heads, self.rope_dim)

        # Step 5: Apply partial RoPE and concatenate
        queries_full = self._apply_partial_rope(
            queries_content, queries_rope, self.query_rope_module
        )
        keys_full = self._apply_partial_rope(keys_content, keys_rope, self.key_rope_module)

        # Step 6: Transpose to [batch, heads, seq, dim] for attention
        queries_full = queries_full.transpose(1, 2)
        keys_full = keys_full.transpose(1, 2)
        values = values.transpose(1, 2)

        # Step 7: Compute attention using FlashMLA or manual attention
        total_dim = self.head_dim + self.rope_dim

        # Get mscale factor from key RoPE module (only keys have YaRN)
        # This will be 1.0 for vanilla RoPE, or the YaRN mscale for extended context
        mscale = self.key_rope_module.get_attention_scale_factor()

        # Compute attention scale with mscale adjustment for YaRN
        attention_scale = (1.0 / math.sqrt(total_dim)) * mscale

        attn_output: jaxtyping.Float[torch.Tensor, "batch heads seq head_dim"]

        if self.use_flash_attention:
            # Use FlashMLA optimized kernels
            attn_output = self._compute_flash_mla_attention(
                queries_full,
                keys_full,
                values,
                total_dim,
                self.is_causal,
                attention_scale,
                self.attn_dropout,
            )
        else:
            # Use manual attention computation
            attn_output = self._compute_manual_attention(
                queries_full,
                keys_full,
                values,
                total_dim,  # This is the key! MLA uses total_dim for scaling
                self.is_causal,
                attention_scale,  # Now includes mscale adjustment
                attention_mask,
                None,  # No causal mask buffer
                self.attn_dropout,
            )

        # Step 8: Merge heads back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Step 9: Apply output projection
        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self.output_proj(attn_output)

        return output
