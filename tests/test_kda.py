"""Check the CPU KDA against fla's reference implementations.

fla-core's ops/kda/naive.py (pure PyTorch) is loaded directly as a file: importing the fla package
would import Triton.
"""

import importlib.util
from importlib import metadata
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from haas_asr_cpu.kda import KimiDeltaAttentionCPU, ShortConvolutionCPU, recurrent_kda


def _fla_file(rel: str) -> Path:
    files = metadata.files("fla-core") or []
    for f in files:
        if str(f).replace("\\", "/") == rel:
            return Path(f.locate())
    pytest.skip(f"{rel} not found in fla-core")


def _load(rel: str):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), _fla_file(rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def naive():
    return _load("fla/ops/kda/naive.py")


def _inputs(B=2, T=64, H=5, K=64, V=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, K, generator=g), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, generator=g), dim=-1)
    v = torch.randn(B, T, H, V, generator=g)
    gate = -F.softplus(torch.randn(B, T, H, K, generator=g))  # log-space decay (< 0)
    beta = torch.rand(B, T, H, generator=g)
    s0 = torch.randn(B, H, K, V, generator=g) * 0.1
    return q, k, v, gate, beta, s0


def test_recurrent_matches_fla_naive_recurrent(naive):
    q, k, v, g, beta, s0 = _inputs()
    o_ref, s_ref = naive.naive_recurrent_kda(q, k, v, g, beta, initial_state=s0, output_final_state=True)
    o, s = recurrent_kda(q, k, v, g, beta, scale=64 ** -0.5, initial_state=s0)
    torch.testing.assert_close(o, o_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s, s_ref, rtol=1e-5, atol=1e-5)


def test_recurrent_matches_fla_naive_chunk(naive):
    """Also matches a different algorithm (chunk-parallel), so the recurrence was copied correctly."""
    q, k, v, g, beta, s0 = _inputs(T=128)
    o_ref, s_ref = naive.naive_chunk_kda(q, k, v, g, beta, initial_state=s0, output_final_state=True)
    o, s = recurrent_kda(q, k, v, g, beta, scale=64 ** -0.5, initial_state=s0)
    torch.testing.assert_close(o, o_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(s, s_ref, rtol=1e-4, atol=1e-4)


def test_short_conv_step_equals_full():
    torch.manual_seed(0)
    conv = ShortConvolutionCPU(16, 4)
    x = torch.randn(2, 10, 16)
    full, cache_full = conv(x, output_final_state=True)
    cache, outs = None, []
    for t in range(10):
        y, cache = conv(x[:, t:t + 1], cache=cache, output_final_state=True)
        outs.append(y)
    torch.testing.assert_close(torch.cat(outs, 1), full)
    torch.testing.assert_close(cache, cache_full)
    # causal: changing later inputs does not change earlier outputs
    x2 = x.clone()
    x2[:, 5:] += 1
    torch.testing.assert_close(conv(x2)[0][:, :5], full[:, :5])


class _Cache:
    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        return self.states[i]

    def update(self, layer_idx=0, offset=None, **state):
        while len(self.states) <= layer_idx:
            self.states.append({})
        self.states[layer_idx].update(state)


def test_layer_step_by_step_equals_full_sequence():
    """Decoding one token at a time with a cache gives the same output as the full sequence."""
    torch.manual_seed(0)
    layer = KimiDeltaAttentionCPU(hidden_size=64, head_dim=16, num_heads=4, layer_idx=0).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0, 0.2)
        x = torch.randn(3, 12, 64)
        full, _, _ = layer(x, past_key_values=_Cache(), use_cache=True)
        cache, outs = _Cache(), []
        for t in range(12):
            y, _, _ = layer(x[:, t:t + 1], past_key_values=cache, use_cache=True)
            outs.append(y)
    torch.testing.assert_close(torch.cat(outs, 1), full, rtol=1e-5, atol=1e-5)


def test_unsupported_arguments_fail_clearly():
    with pytest.raises(ValueError):
        KimiDeltaAttentionCPU(hidden_size=64, head_dim=16, num_heads=4, mode="bogus")
    with pytest.raises(ValueError):
        KimiDeltaAttentionCPU(hidden_size=64, head_dim=16, num_heads=4, num_v_heads=6)
    for kwargs in ({"use_short_conv": False}, {"lower_bound": -5.0}, {"allow_neg_eigval": True}):
        with pytest.raises(NotImplementedError):
            KimiDeltaAttentionCPU(hidden_size=64, head_dim=16, num_heads=4, **kwargs)
    layer = KimiDeltaAttentionCPU(hidden_size=64, head_dim=16, num_heads=4, layer_idx=0)
    with pytest.raises(NotImplementedError):
        layer(torch.randn(1, 4, 64), cu_seqlens=torch.tensor([0, 2, 4]))
    with pytest.raises(NotImplementedError):
        layer(torch.randn(1, 4, 64), attention_mask=torch.ones(1, 4))


def test_no_state_kept_without_use_cache():
    layer = KimiDeltaAttentionCPU(hidden_size=64, head_dim=16, num_heads=4, layer_idx=0)
    cache = _Cache()
    with torch.no_grad():
        layer(torch.randn(1, 3, 64), past_key_values=cache, use_cache=False)
    assert cache[0]["recurrent_state"] is None and cache[0]["conv_state"] is None


def test_expand_v_tolerance_as_fla():
    assert KimiDeltaAttentionCPU(hidden_size=64, head_dim=100, num_heads=1, expand_v=0.07).head_v_dim == 7
