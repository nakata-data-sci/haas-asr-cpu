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
    assert cache[0]["recurrent_state"] is None and cache[0]["conv_state"] == (None, None, None)  # as fla


def test_expand_v_tolerance_as_fla():
    assert KimiDeltaAttentionCPU(hidden_size=64, head_dim=100, num_heads=1, expand_v=0.07).head_v_dim == 7
    # 100 * 0.0700001 = 7.00001: within fla's relative tolerance of 1e-5
    assert KimiDeltaAttentionCPU(hidden_size=64, head_dim=100, num_heads=1, expand_v=0.0700001).head_v_dim == 7
    with pytest.raises(ValueError):  # 15 * 0.5 = 7.5 is not an integer
        KimiDeltaAttentionCPU(hidden_size=64, head_dim=15, num_heads=1, expand_v=0.5)


def test_recurrent_matches_fla_with_grouped_value_heads(naive):
    g = torch.Generator().manual_seed(1)
    B, T, H, HV, K, V = 2, 32, 2, 4, 16, 16
    q = F.normalize(torch.randn(B, T, H, K, generator=g), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, generator=g), dim=-1)
    v = torch.randn(B, T, HV, V, generator=g)
    gate = -F.softplus(torch.randn(B, T, HV, K, generator=g))
    beta = torch.rand(B, T, HV, generator=g)
    o_ref, s_ref = naive.naive_recurrent_kda(q, k, v, gate, beta, output_final_state=True)
    o, s = recurrent_kda(q, k, v, gate, beta, scale=K ** -0.5)
    torch.testing.assert_close(o, o_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s, s_ref, rtol=1e-5, atol=1e-5)


def test_install_makes_haas_build_the_cpu_layer():
    import haas_asr.prednet as prednet

    from haas_asr_cpu.kda import install

    install()
    assert prednet.load_kda_class() is KimiDeltaAttentionCPU


def test_layer_scales_queries_by_head_dim_as_fla(monkeypatch):
    import haas_asr_cpu.kda as kda

    seen = []
    real = kda.recurrent_kda
    monkeypatch.setattr(kda, "recurrent_kda", lambda *a, scale, **k: seen.append(scale) or real(*a, scale=scale, **k))
    layer = KimiDeltaAttentionCPU(hidden_size=64, head_dim=16, num_heads=4, expand_v=2)  # head_v_dim = 32
    with torch.no_grad():
        layer(torch.randn(1, 3, 64))
    assert seen == [16 ** -0.5]  # fla's default scale is head_k_dim ** -0.5, not head_v_dim


def test_parameter_names_and_shapes_match_the_released_checkpoint():
    # Haas-ja-v1.0's prediction network layer: hidden 640, 5 heads of 64, conv 4 (names as in fla 0.5.2)
    layer = KimiDeltaAttentionCPU(hidden_size=640, head_dim=64, num_heads=5, conv_size=4)
    assert {k: tuple(v.shape) for k, v in layer.state_dict().items()} == {
        "A_log": (5,), "dt_bias": (320,), "b_proj.weight": (5, 640),
        "q_proj.weight": (320, 640), "k_proj.weight": (320, 640), "v_proj.weight": (320, 640),
        "q_conv1d.weight": (320, 1, 4), "k_conv1d.weight": (320, 1, 4), "v_conv1d.weight": (320, 1, 4),
        "f_proj.0.weight": (64, 640), "f_proj.1.weight": (320, 64),
        "g_proj.0.weight": (64, 640), "g_proj.1.weight": (320, 64), "g_proj.1.bias": (320,),
        "o_norm.weight": (64,), "o_proj.weight": (640, 320),
    }


def test_haas_prediction_network_with_the_cpu_layer_decodes_step_by_step():
    """Haas's own KDAPredictionNetwork, built after install(), packs the layer's states into NeMo's two
    tensors; decoding one label at a time through that packing equals the full sequence."""
    from haas_asr.prednet import KDAConfig, KDAPredictionNetwork

    from haas_asr_cpu.kda import install

    install()
    torch.manual_seed(0)
    net = KDAPredictionNetwork(64, 2, KDAConfig(num_heads=2, head_dim=16, expand_v=2.0, conv_size=4,
                                                final_norm=True)).eval()
    assert all(isinstance(layer, KimiDeltaAttentionCPU) for layer in net.layers)
    y = torch.randn(7, 3, 64)  # (U, B, H), time first as NeMo calls it
    with torch.no_grad():
        full, _ = net(y)
        state, outs = None, []
        for u in range(7):
            out, state = net(y[u:u + 1], state)
            outs.append(out)
    torch.testing.assert_close(torch.cat(outs, 0), full, rtol=1e-5, atol=1e-5)


def _reference_layer(layer, x, naive):
    """fla 0.5.2 KimiDeltaAttention.forward (inference, no cache) written out independently of kda.py.

    short conv + SiLU: modules/conv/short_conv.py / gate -exp(A_log) * softplus(g + dt_bias): ops/kda/gate.py /
    q, k l2-normalised with eps 1e-6 and sigmoid on beta: ops/kda/fused_recurrent.py / recurrence: fla's own
    naive_recurrent_kda / RMSNorm (eps 1e-5) * weight * sigmoid(g): modules/fused_norm_gate.py.
    """
    B, T, _ = x.shape
    H, HV, K, V = layer.num_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim

    def conv(m, u):  # causal depthwise convolution over time
        y = F.conv1d(F.pad(u.transpose(1, 2), (m.kernel_size[0] - 1, 0)), m.weight, m.bias, groups=u.shape[-1])
        return F.silu(y).transpose(1, 2)

    def l2(t):
        return t * torch.rsqrt(t.pow(2).sum(-1, keepdim=True) + 1e-6)

    q = l2(conv(layer.q_conv1d, layer.q_proj(x)).reshape(B, T, H, K))
    k = l2(conv(layer.k_conv1d, layer.k_proj(x)).reshape(B, T, H, K))
    v = conv(layer.v_conv1d, layer.v_proj(x)).reshape(B, T, HV, V)
    g = layer.f_proj(x).reshape(B, T, HV, K)
    g = -torch.exp(layer.A_log)[:, None] * F.softplus(g + layer.dt_bias.reshape(HV, K))
    beta = torch.sigmoid(layer.b_proj(x))
    o, _ = naive.naive_recurrent_kda(q, k, v, g, beta, scale=K ** -0.5)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-5) * layer.o_norm.weight
    o = o * torch.sigmoid(layer.g_proj(x).reshape(B, T, HV, V))
    return layer.o_proj(o.reshape(B, T, HV * V))


def test_whole_layer_matches_fla_formulas(naive):
    """The gate, beta, l2 norm, short conv and gated norm, not only the recurrence, follow fla."""
    torch.manual_seed(3)
    layer = KimiDeltaAttentionCPU(hidden_size=32, head_dim=8, num_heads=2, num_v_heads=4, expand_v=2).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0, 0.3)
        layer.A_log.normal_(0, 0.5)
        layer.dt_bias.normal_(0, 1.0)
        x = torch.randn(2, 10, 32) * 0.1  # small inputs, so both eps values matter
        out, _, _ = layer(x)
        ref = _reference_layer(layer, x, naive)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_parameter_shapes_with_grouped_heads_expand_v_and_conv_bias():
    """Shapes from fla 0.5.2 layers/kda.py __init__ for a config other than Haas's (K != V, HV != H, bias)."""
    layer = KimiDeltaAttentionCPU(hidden_size=32, head_dim=8, num_heads=2, num_v_heads=4, expand_v=2,
                                  conv_bias=True)
    # K = 8, V = 16, key_dim = 2*8 = 16, value_dim = 4*16 = 64, gate_dim = 4*8 = 32
    assert {k: tuple(v.shape) for k, v in layer.state_dict().items()} == {
        "A_log": (4,), "dt_bias": (32,), "b_proj.weight": (4, 32),
        "q_proj.weight": (16, 32), "k_proj.weight": (16, 32), "v_proj.weight": (64, 32),
        "q_conv1d.weight": (16, 1, 4), "q_conv1d.bias": (16,), "k_conv1d.weight": (16, 1, 4),
        "k_conv1d.bias": (16,), "v_conv1d.weight": (64, 1, 4), "v_conv1d.bias": (64,),
        "f_proj.0.weight": (16, 32), "f_proj.1.weight": (32, 16),
        "g_proj.0.weight": (16, 32), "g_proj.1.weight": (64, 16), "g_proj.1.bias": (64,),
        "o_norm.weight": (16,), "o_proj.weight": (32, 64),
    }


def test_whole_layer_with_conv_bias_matches_fla_formulas(naive):
    torch.manual_seed(4)
    layer = KimiDeltaAttentionCPU(hidden_size=32, head_dim=8, num_heads=2, conv_bias=True).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0, 0.3)
        x = torch.randn(1, 6, 32) * 0.1
        out, _, _ = layer(x)
        ref = _reference_layer(layer, x, naive)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_recurrent_matches_fla_with_slow_decay(naive):
    """Per-step decays close to 1 (long memory), as fla initialises them, not only 0.5-0.95."""
    q, k, v, _, beta, s0 = _inputs(T=128, seed=5)
    gate = -torch.rand(q.shape, generator=torch.Generator().manual_seed(6)) * 1e-3  # decay 0.999-1.0
    o_ref, s_ref = naive.naive_recurrent_kda(q, k, v, gate, beta, initial_state=s0, output_final_state=True)
    o, s = recurrent_kda(q, k, v, gate, beta, scale=q.shape[-1] ** -0.5, initial_state=s0)
    torch.testing.assert_close(o, o_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s, s_ref, rtol=1e-5, atol=1e-5)


def test_cache_without_layer_idx_fails_clearly():
    layer = KimiDeltaAttentionCPU(hidden_size=64, head_dim=16, num_heads=4)
    with pytest.raises(ValueError, match="layer_idx"):
        layer(torch.randn(1, 2, 64), past_key_values=_Cache(), use_cache=True)
