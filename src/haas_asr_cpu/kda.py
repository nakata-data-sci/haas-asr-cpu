"""A pure-PyTorch drop-in for fla's KimiDeltaAttention, so Haas-ja runs without CUDA.

The prediction network of Haas-ja-v1.0 (2 layers, 3.9M parameters) uses KimiDeltaAttention from
flash-linear-attention (fla) 0.5.2, whose inference kernels need Triton + CUDA. This module re-implements
the same layer with the same parameter names, so the released weights load unchanged. The math follows
fla-core 0.5.2 (MIT License, see NOTICE):
  - ops/kda/naive.py `naive_recurrent_kda` (state update and output)
  - ops/kda/gate.py `naive_kda_gate` (g = -exp(A_log) * softplus(g + dt_bias))
  - ops/kda/fused_recurrent.py (L2-normalised q and k with eps=1e-6, sigmoid on beta)
  - modules/conv/short_conv.py `step` (causal depthwise conv + SiLU; the cache holds the last W inputs)
  - modules/fused_norm_gate.py (RMSNorm * weight * sigmoid(g))

The recurrent state is laid out as [B, HV, K, V], unlike fla's state_v_first. The state never leaves this
layer (Haas's prediction network only reshapes and carries it), so the results are the same.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ShortConvolutionCPU(nn.Conv1d):
    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False,
                 activation: str | None = "silu", **_):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.activation = activation

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None = None,
                output_final_state: bool = False, **_):
        """x: [B, T, D] -> y: [B, T, D]. cache: [B, D, W], the last W inputs (zeros at the start)."""
        B, T, D = x.shape
        W = self.kernel_size[0]
        prev = cache if cache is not None else x.new_zeros(B, D, W)
        full = torch.cat([prev[..., 1:].to(x.dtype), x.transpose(1, 2)], dim=-1)  # [B, D, W-1+T]
        y = F.conv1d(full, self.weight, self.bias, groups=D)  # [B, D, T]
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        new_cache = full[..., -W:].contiguous() if output_final_state else None
        return y.transpose(1, 2), new_cache


class RMSNormGatedCPU(nn.Module):
    def __init__(self, hidden_size: int, activation: str = "sigmoid", eps: float = 1e-5, **_):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.activation = activation
        self.eps = eps

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x, g = x.float(), g.float()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        if self.activation == "sigmoid":
            y = y * torch.sigmoid(g)
        else:  # swish / silu
            y = y * g * torch.sigmoid(g)
        return y.to(dtype)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)


def recurrent_kda(q, k, v, g, beta, scale, initial_state=None):
    """Same as fla `naive_recurrent_kda`. q, k: [B,T,H,K] / v: [B,T,HV,V] / g: [B,T,HV,K] / beta: [B,T,HV]."""
    dtype = v.dtype
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // H
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state.float()
    o = torch.empty_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum("bhk,bhv->bhkv", b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)
    return o.to(dtype), S


class KimiDeltaAttentionCPU(nn.Module):
    """Same arguments and parameter names as fla.layers.kda.KimiDeltaAttention (0.5.2)."""

    def __init__(self, hidden_size: int = 2048, expand_v: float = 1, head_dim: int = 128,
                 num_heads: int = 16, num_v_heads: int | None = None, mode: str = "chunk",
                 use_short_conv: bool = True, allow_neg_eigval: bool = False,
                 safe_gate: bool = False, lower_bound: float | None = None, conv_size: int = 4,
                 conv_bias: bool = False, layer_idx: int | None = None, norm_eps: float = 1e-5, **_):
        super().__init__()
        if not use_short_conv:
            raise NotImplementedError("use_short_conv=False is not used by Haas")
        if lower_bound is not None or allow_neg_eigval:
            raise NotImplementedError("lower_bound / allow_neg_eigval are not used by Haas")
        if mode not in ("chunk", "fused_recurrent"):
            raise ValueError(f"mode must be 'chunk' or 'fused_recurrent', got {mode!r}")  # as fla
        if num_v_heads is not None and num_v_heads % num_heads != 0:
            raise ValueError(f"num_v_heads ({num_v_heads}) must be a multiple of num_heads ({num_heads})")
        if not math.isclose(head_dim * expand_v, int(head_dim * expand_v), rel_tol=1e-5):  # as fla
            raise ValueError(f"head_dim * expand_v ({head_dim * expand_v}) must be an integer")
        self.mode = mode
        self.hidden_size = hidden_size
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_conv1d = ShortConvolutionCPU(self.key_dim, conv_size, bias=conv_bias)
        self.k_conv1d = ShortConvolutionCPU(self.key_dim, conv_size, bias=conv_bias)
        self.v_conv1d = ShortConvolutionCPU(self.value_dim, conv_size, bias=conv_bias)
        self.gate_dim = int(self.num_v_heads * self.head_k_dim)
        self.f_proj = nn.Sequential(nn.Linear(hidden_size, self.head_v_dim, bias=False),
                                    nn.Linear(self.head_v_dim, self.gate_dim, bias=False))
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.gate_dim))
        self.g_proj = nn.Sequential(nn.Linear(hidden_size, self.head_v_dim, bias=False),
                                    nn.Linear(self.head_v_dim, self.value_dim, bias=True))
        self.o_norm = RMSNormGatedCPU(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, past_key_values=None,
                use_cache: bool | None = False, output_attentions: bool | None = False, **kwargs):
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not used by Haas inference")
        if kwargs.get("cu_seqlens") is not None:
            raise NotImplementedError("packed sequences (cu_seqlens) are not used by Haas inference")
        if past_key_values is not None and self.layer_idx is None:
            raise ValueError("layer_idx is required when past_key_values is given")
        x = hidden_states
        B, T, _ = x.shape
        last = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last = past_key_values[self.layer_idx]
        cq = ck = cv = None
        if last is not None and last.get("conv_state") is not None:
            cq, ck, cv = last["conv_state"]
        q, cq = self.q_conv1d(self.q_proj(x), cache=cq, output_final_state=use_cache)
        k, ck = self.k_conv1d(self.k_proj(x), cache=ck, output_final_state=use_cache)
        v, cv = self.v_conv1d(self.v_proj(x), cache=cv, output_final_state=use_cache)

        HV, K, V = self.num_v_heads, self.head_k_dim, self.head_v_dim
        q = _l2norm(q.view(B, T, self.num_heads, K).float())
        k = _l2norm(k.view(B, T, self.num_heads, K).float())
        v = v.view(B, T, HV, V)
        g = self.f_proj(x).view(B, T, HV, K).float()
        g = -self.A_log.float().exp().view(HV, 1) * F.softplus(g + self.dt_bias.float().view(HV, K))
        beta = torch.sigmoid(self.b_proj(x).float())

        S0 = last.get("recurrent_state") if last is not None else None
        o, S = recurrent_kda(q, k, v, g, beta, scale=K ** -0.5, initial_state=S0)

        if past_key_values is not None:
            # as fla: the states are kept only with use_cache
            past_key_values.update(layer_idx=self.layer_idx, recurrent_state=S if use_cache else None,
                                   conv_state=(cq, ck, cv), offset=T)  # each None without use_cache
        o = self.o_norm(o, self.g_proj(x).view(B, T, HV, V))
        o = self.o_proj(o.reshape(B, T, self.value_dim).to(x.dtype))
        return o, None, past_key_values


def install() -> None:
    """Make haas_asr build its prediction network with this layer instead of fla. Call before loading a model.

    The replacement stays for the rest of the process: any later haas_asr.load_model() also gets this layer.
    """
    import haas_asr.prednet as prednet

    prednet.load_kda_class = lambda: KimiDeltaAttentionCPU
