# Third Party
import jaxtyping
import torch


class MultiHeadReshapeMixin:
    """
    Mixin for reshaping tensors between multi-head and standard formats.

    Scholarship:
    Attention Is All You Need, 2017, https://arxiv.org/pdf/1706.03762, 3.2.2

    Significance:
    Allows model to attend to different representation subspaces simultaneously.
    Each head can learn different types of relationships between tokens.

    Init:
    This mixin has no initialization or learnable parameters.
    It provides tensor reshaping utilities for multi-head attention.

    Step-by-step control flow (_reshape_to_multihead):
    1. Receive flat tensor [batch, seq, hidden_dim]
    2. Reshape to separate heads [batch, seq, n_heads, head_dim]
    3. Transpose to attention format [batch, n_heads, seq, head_dim]

    Step-by-step control flow (_merge_heads):
    1. Receive multi-head tensor [batch, n_heads, seq, head_dim]
    2. Transpose back [batch, seq, n_heads, head_dim]
    3. Flatten heads dimension [batch, seq, hidden_dim]

    Learning/Optimization process:
    - This mixin contains no learnable parameters.
    - It performs pure tensor reshaping operations.

    Purpose in architecture:
    - Splitting enables each head to specialize in different patterns
    - Parallel heads prevent averaging that would inhibit diverse attention patterns
    - Merging combines all heads' outputs for the output projection to process
    """

    def _reshape_to_multihead(
        self,
        tensor: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_len head_dim"]:
        """
        Reshape tensor for multi-head attention computation.
        """
        # First reshape to separate head dimension
        reshaped: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads head_dim"]
        reshaped = tensor.view(batch_size, seq_len, num_heads, head_dim)

        # Transpose to put heads before sequence
        multihead: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        multihead = reshaped.transpose(1, 2)

        return multihead

    def _merge_heads(
        self,
        tensor: jaxtyping.Float[torch.Tensor, "batch n_heads seq_len head_dim"],
        hidden_dim: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Merge attention heads back to single tensor.
        """
        batch_size, num_heads, seq_len, head_dim = tensor.shape

        # Transpose heads and sequence dimensions
        x_transposed: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads head_dim"]
        x_transposed = tensor.transpose(1, 2).contiguous()

        # Reshape to combine heads
        x_merged: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        x_merged = x_transposed.view(batch_size, seq_len, hidden_dim)

        return x_merged
