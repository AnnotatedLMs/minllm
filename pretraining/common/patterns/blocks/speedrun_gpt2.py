# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import flex_attention

# Project
from pretraining.common.patterns.attention import normalized_flex_attention
from pretraining.common.patterns.ffn import mlp
from pretraining.common.patterns.position import rope_truncated


class SpeedrunGPT2Block(nn.Module):
    """
    Speedrun-optimized GPT-2 transformer block matching nanogpt.

    Combines all speedrun optimizations:
    - NormalizedFlexAttention with QK norm
    - Half-truncated RoPE
    - ReLU-squared activation in MLP
    - Value embedding mixing (handled at block level)
    - Conditional layer skipping (layer 7 has no attention)
    - Custom attention scale

    The block orchestrates how these components work together,
    handling value embeddings and skip connections at the appropriate level.
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_dim: int,
        num_heads: int,
        head_dim: int = 128,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        attention_scale: float = 0.12,  # scale the attention logits by given constant @leloykun
        block_size: int = 128,
        eot_token_id: int = 50256,
    ):
        """
        Initialize SpeedrunGPT2Block.

        Args:
            layer_idx: Layer index (for conditional skipping)
            hidden_dim: Model dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head (default 128 from nanogpt)
            max_seq_len: Maximum sequence length for RoPE
            dropout: Dropout probability
            norm_eps: Epsilon for layer normalization
            attention_scale: Custom attention scale (default 0.12 from nanogpt)
            block_size: Block size for FlexAttention
            eot_token_id: End-of-text token for document boundaries
        """
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.eot_token_id = eot_token_id

        # Layer normalization (no bias for speedrun)
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=norm_eps, bias=False)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=norm_eps, bias=False)

        # Conditional attention (skip layer 7)
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        if layer_idx != 7:
            self.attention = normalized_flex_attention.NormalizedFlexAttention(
                hidden_dim=num_heads * head_dim,  # Note: using explicit head_dim
                num_heads=num_heads,
                dropout=dropout,
                bias=False,  # No bias in speedrun
                max_seq_length=max_seq_len,
                is_causal=True,
                attention_scale=attention_scale,
                block_size=block_size,
                norm_eps=norm_eps,
            )

            # Half-truncated RoPE for this attention layer
            self.rope = rope_truncated.HalfTruncatedRoPE(
                dim=head_dim,
                theta=1024.0,
                max_seq_len=max_seq_len,
            )
        else:
            self.attention = None
            self.rope = None

        # MLP with ReLU-squared
        self.mlp = mlp.MLP(
            hidden_dim=hidden_dim,
            intermediate_dim=4 * hidden_dim,
            dropout=dropout,
            activation="relu_squared",  # ~1-2% better than GELU
            bias=False,  # No bias in speedrun
        )

    def _create_document_block_mask(
        self,
        input_ids: jaxtyping.Int[torch.Tensor, "seq"],
        sliding_window_blocks: typing.Optional[int] = None,
    ) -> flex_attention.BlockMask:
        """
        Create document-aware block mask for FlexAttention.

        Documents are separated by EOT tokens.
        """
        seq_len = input_ids.shape[0]
        docs = (input_ids == self.eot_token_id).cumsum(0)

        if sliding_window_blocks is not None:
            # Sliding window with document boundaries
            window_size = sliding_window_blocks * 128  # block_size

            def mask_fn(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                same_doc = docs[q_idx] == docs[kv_idx]
                in_window = (q_idx - kv_idx) <= window_size
                return causal & same_doc & in_window
        else:
            # Just document-aware causal
            def mask_fn(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                same_doc = docs[q_idx] == docs[kv_idx]
                return causal & same_doc

        # Create BlockMask
        block_mask = flex_attention.create_block_mask(
            mask_fn,
            B=1,  # Batch size (nanogpt uses B=1)
            H=self.attention.num_heads if self.attention else 1,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            BLOCK_SIZE=128,
        )

        return block_mask

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq dim"],
        value_embed: typing.Optional[jaxtyping.Float[torch.Tensor, "seq dim"]] = None,
        x0: typing.Optional[jaxtyping.Float[torch.Tensor, "batch seq dim"]] = None,
        lambdas: typing.Optional[jaxtyping.Float[torch.Tensor, "2"]] = None,
        sa_lambdas: typing.Optional[jaxtyping.Float[torch.Tensor, "2"]] = None,
        block_mask: typing.Optional[flex_attention.BlockMask] = None,
        input_ids: typing.Optional[jaxtyping.Int[torch.Tensor, "seq"]] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq dim"]:
        """
        Forward pass through SpeedrunGPT2Block.

        Matches nanogpt's Block.forward exactly:
        1. Mix residual with x0 if provided (for U-Net skip)
        2. Apply attention with value embedding mixing
        3. Apply MLP

        Args:
            x: Input tensor [batch, seq, dim]
            value_embed: Optional value embeddings [seq, dim] from model
            x0: Optional original input for residual mixing (U-Net)
            lambdas: Mixing weights for x and x0
            sa_lambdas: Mixing weights for values in attention
            block_mask: Pre-computed BlockMask for FlexAttention
            input_ids: Token IDs for creating document mask (if block_mask not provided)

        Returns:
            Output tensor [batch, seq, dim]
        """
        # Mix with original input if provided (U-Net skip connections)
        if x0 is not None and lambdas is not None:
            x = lambdas[0] * x + lambdas[1] * x0

        # Attention sublayer (if not skipped)
        if self.attention is not None:
            # Normalize input
            x_norm = self.attention_norm(x)

            # Create block mask if needed
            if block_mask is None and input_ids is not None:
                block_mask = self._create_document_block_mask(input_ids)

            # Apply attention with RoPE
            batch_size, seq_len, _ = x_norm.shape

            # Get Q, K, V projections
            q, k, v = self.attention._compute_qkv_projections(x_norm)

            # Reshape to multi-head format [batch, seq, heads, head_dim]
            q_heads = q.view(batch_size, seq_len, self.attention.num_heads, -1)
            k_heads = k.view(batch_size, seq_len, self.attention.num_heads, -1)
            v_heads = v.view(batch_size, seq_len, self.attention.num_heads, -1)

            # Apply RoPE to Q and K
            q_heads = self.rope(q_heads)
            k_heads = self.rope(k_heads)

            # Mix value embeddings if provided
            if value_embed is not None and sa_lambdas is not None:
                # Reshape value embed to match v_heads shape
                ve_heads = value_embed.view(seq_len, self.attention.num_heads, -1)
                ve_heads = ve_heads.unsqueeze(0).expand(batch_size, -1, -1, -1)
                # Mix: v = lambda[0] * v + lambda[1] * ve
                v_heads = sa_lambdas[0] * v_heads + sa_lambdas[1] * ve_heads
            elif sa_lambdas is not None:
                # Scale values only (for layers without value embeddings)
                v_heads = sa_lambdas[0] * v_heads

            # Transpose to [batch, heads, seq, head_dim] for attention
            q_heads = q_heads.transpose(1, 2)
            k_heads = k_heads.transpose(1, 2)
            v_heads = v_heads.transpose(1, 2)

            # Apply QK norm
            q_heads = F.rms_norm(q_heads, (q_heads.size(-1),))
            k_heads = F.rms_norm(k_heads, (k_heads.size(-1),))

            # Apply FlexAttention
            attn_out = self.attention._compute_flex_attention(q_heads, k_heads, v_heads, block_mask)

            # Merge heads and project
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.view(batch_size, seq_len, -1)
            attn_out = self.attention._apply_output_projection(attn_out)

            # Add residual
            x = x + attn_out

        # FFN sublayer
        x_norm = self.ffn_norm(x)
        ffn_out = self.mlp(x_norm)
        x = x + ffn_out

        return x
