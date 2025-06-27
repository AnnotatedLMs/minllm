# Standard Library
import inspect
import math
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn
import transformers
from torch.nn import functional as F

# Project
from pretraining.gpt.models import config


class LayerNorm(nn.Module):
    """LayerNorm with optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(
        self, input: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: config.GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention support (PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask to ensure attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv: jaxtyping.Float[torch.Tensor, "batch seq 3*d_model"] = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k: jaxtyping.Float[torch.Tensor, "batch n_head seq head_dim"] = k.view(
            B, T, self.n_head, C // self.n_head
        ).transpose(1, 2)
        q: jaxtyping.Float[torch.Tensor, "batch n_head seq head_dim"] = q.view(
            B, T, self.n_head, C // self.n_head
        ).transpose(1, 2)
        v: jaxtyping.Float[torch.Tensor, "batch n_head seq head_dim"] = v.view(
            B, T, self.n_head, C // self.n_head
        ).transpose(1, 2)

        # Causal self-attention
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y: jaxtyping.Float[torch.Tensor, "batch n_head seq head_dim"] = (
                torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True,
                )
            )
        else:
            # Manual implementation of attention
            att: jaxtyping.Float[torch.Tensor, "batch n_head seq seq"] = (
                q @ k.transpose(-2, -1)
            ) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y: jaxtyping.Float[torch.Tensor, "batch n_head seq head_dim"] = att @ v

        # Re-assemble all head outputs side by side
        y: jaxtyping.Float[torch.Tensor, "batch seq d_model"] = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: config.GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        # Expand to 4x hidden dimension
        x: jaxtyping.Float[torch.Tensor, "batch seq 4*d_model"] = self.c_fc(x)
        x = self.gelu(x)
        # Project back to d_model
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"] = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: config.GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        n_layer: typing.Optional[int] = None,
        n_head: typing.Optional[int] = None,
        n_embd: typing.Optional[int] = None,
        vocab_size: typing.Optional[int] = None,
        block_size: typing.Optional[int] = None,
        dropout: typing.Optional[float] = None,
        bias: typing.Optional[bool] = None,
        model_conf: typing.Optional[config.GPTConfig] = None,
    ):
        super().__init__()

        # Handle both config object and individual parameters
        if model_conf is not None:
            self.config = model_conf
        else:
            # Build config from individual parameters
            self.config = config.GPTConfig(
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                vocab_size=vocab_size,
                block_size=block_size,
                dropout=dropout if dropout is not None else 0.0,
                bias=bias if bias is not None else True,
            )

        assert self.config.vocab_size is not None
        assert self.config.block_size is not None

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.config.vocab_size, self.config.n_embd),
                wpe=nn.Embedding(self.config.block_size, self.config.n_embd),
                drop=nn.Dropout(self.config.dropout),
                h=nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
                ln_f=LayerNorm(self.config.n_embd, bias=self.config.bias),
            )
        )
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: jaxtyping.Int[torch.Tensor, "batch seq"],
        targets: typing.Optional[jaxtyping.Int[torch.Tensor, "batch seq"]] = None,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq vocab"],
        typing.Optional[jaxtyping.Float[torch.Tensor, ""]],
    ]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos: jaxtyping.Int[torch.Tensor, "seq"] = torch.arange(
            0, t, dtype=torch.long, device=device
        )

        # Forward the GPT model
        tok_emb: jaxtyping.Float[torch.Tensor, "batch seq d_model"] = self.transformer.wte(idx)
        pos_emb: jaxtyping.Float[torch.Tensor, "seq d_model"] = self.transformer.wpe(pos)
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"] = self.transformer.drop(
            tok_emb + pos_emb
        )

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"] = self.lm_head(x)
            loss: jaxtyping.Float[torch.Tensor, ""] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits: jaxtyping.Float[torch.Tensor, "batch 1 vocab"] = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        # Model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type: str) -> "GPT":
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints

        # Create a from-scratch initialized minGPT model
        model_conf = config.GPTConfig(**config_args)
        model = GPT(model_conf=model_conf)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        model_hf = transformers.GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # OpenAI checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: typing.Tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        # First estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        idx: jaxtyping.Int[torch.Tensor, "batch seq"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: typing.Optional[int] = None,
    ) -> jaxtyping.Int[torch.Tensor, "batch seq"]:
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits: jaxtyping.Float[torch.Tensor, "batch vocab"] = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Apply softmax to convert logits to (normalized) probabilities
            probs: jaxtyping.Float[torch.Tensor, "batch vocab"] = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next: jaxtyping.Int[torch.Tensor, "batch 1"] = torch.multinomial(
                probs, num_samples=1
            )
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
