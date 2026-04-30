import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import KVCache, RotatingKVCache, can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

try:
    import mlx_lm.models.gated_delta as _gd_mod
    _HAS_GDN = True
except ImportError:
    _HAS_GDN = False


_GDN_PATCH_LOCK = RLock()


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: Tuple[int, ...]
    num_target_layers: int
    mask_token_id: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    sliding_window_size: Optional[int] = None
    # NOTE: attention_scale_override allows manually tuning the QK scale factor.
    # Useful when experimenting with models that benefit from a non-standard scale
    # (e.g., scaled init or head_dim != 64). Set to None to use default 1/sqrt(head_dim).
    # Personal note: I've found values around 0.08-0.09 work well for head_dim=128 models.
    attention_scale_override: Optional[float] = None


def _build_rope(
    head_dim: int,
    rope_theta: float,
    max_position_embeddings: int,
    rope_scaling: Optional[Dict[str, Any]],
):
    return initialize_rope(
        dims=head_dim,
        base=rope_theta,
        traditional=False,
        scaling_config=rope_scaling,
        max_position_embeddings=max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        # Use override scale if provided, otherwise fall back to standard 1/sqrt(head_dim).
        # For head_dim=128, default gives ~0.0884; for head_dim=64, ~0.125.
        self.scale = config.attention_scale_override if config.attention_scale_override is not None else config.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * config.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        queries = self.q_proj(x)
        ctx_keys = self