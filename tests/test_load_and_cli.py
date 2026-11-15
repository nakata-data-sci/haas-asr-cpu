"""load(), transcribe() and the CLI with a stubbed Haas model (no weights are loaded)."""

import importlib
import sys

import haas_asr
import pytest
import torch
from nemo.utils import logging as nemo_logging

t = importlib.import_module("haas_asr_cpu.transcribe")  # the package attribute is the function
cli = importlib.import_module("haas_asr_cpu.cli")


class _Hyp:
    def __init__(self, text, chars):
        self.text = text
        self.timestamp = {"char": [{"char": [c], "start": s, "end": e} for c, s, e in chars]}


class _Model:
    def __init__(self):
        self.calls = []
        self.kwargs = []

    def transcribe(self, chunks, batch_size, **kwargs):
        self.calls.append(batch_size)
        self.kwargs.append(kwargs)
        return [_Hyp("はい。", [("はい。", 0.1, 0.4)]) for _ in chunks]


@pytest.fixture
def restore_process_state():
    level, threads = nemo_logging.getEffectiveLevel(), torch.get_num_threads()
    yield
    nemo_logging.setLevel(level)
    torch.set_num_threads(threads)


@pytest.fixture
def stub_haas(monkeypatch):
    monkeypatch.setattr(haas_asr, "load_audio", lambda *_: [0.0] * 16000)
    monkeypatch.setattr(haas_asr, "split_at_quiet_points", lambda a: [(0, 16000)])
    loaded = []
    monkeypatch.setattr(haas_asr, "load_model", lambda name, device, dtype=None: loaded.append((name, device)) or _Model())
    return loaded


def test_load_installs_the_cpu_layer_quiets_nemo_and_sets_threads(monkeypatch, stub_haas, restore_process_state):
    import haas_asr.prednet as prednet

    monkeypatch.setattr(prednet, "load_kda_class", lambda: "fla")
    nemo_logging.setLevel(nemo_logging.INFO)
    t.load("m", threads=3)
    from haas_asr_cpu.kda import KimiDeltaAttentionCPU

    assert prednet.load_kda_class() is KimiDeltaAttentionCPU and stub_haas == [("m", "cpu")]
    assert nemo_logging.getEffectiveLevel() == nemo_logging.ERROR
    assert torch.get_num_threads() == 3
    nemo_logging.setLevel(nemo_logging.INFO)
    t.load("m", nemo_logs=True)
    assert nemo_logging.getEffectiveLevel() == nemo_logging.INFO


def test_threads_apply_with_a_loaded_model(stub_haas, restore_process_state):
    t.transcribe("x.wav", model=_Model(), threads=2)
    assert torch.get_num_threads() == 2


def test_cli_passes_options_prints_progress_to_stderr_and_q_silences_it(monkeypatch, stub_haas, tmp_path, capsys,
                                                                         restore_process_state):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    model = _Model()
    monkeypatch.setattr(haas_asr, "load_model", lambda name, device, dtype=None: stub_haas.append(name) or model)
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "--model", "my/model", "--threads", "2",
                                      "--batch-size", "3"])
    cli.main()
    out, err = capsys.readouterr()
    assert out == "[00:00:00.100 --> 00:00:00.400] はい。\n"
    assert "[haas-asr-cpu]" in err and "[haas-asr-cpu]" not in out
    assert stub_haas == ["my/model"] and model.calls == [3] and torch.get_num_threads() == 2
    torch.set_num_threads(5)
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q"])
    cli.main()
    assert "[haas-asr-cpu]" not in capsys.readouterr().err
    assert torch.get_num_threads() == 5  # without --threads, torch's setting is left alone


def test_cli_prints_the_transcript_before_writing_files(monkeypatch, stub_haas, tmp_path, capsys,
                                                       restore_process_state):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    from pathlib import Path

    def fail(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q", "--json", str(tmp_path / "o.json")])
    with pytest.raises(OSError):
        cli.main()
    assert "はい。" in capsys.readouterr().out


def test_segments_are_offset_per_chunk_and_clamped(monkeypatch, stub_haas, restore_process_state):
    monkeypatch.setattr(haas_asr, "split_at_quiet_points", lambda a: [(0, 16000), (16000, 20000)])
    model = _Model()
    segs = t.transcribe("x.wav", model=model)
    # chunk 2 starts at 1.0 s and ends at 1.25 s: its token at 0.1-0.4 s becomes 1.1-1.25 s
    assert segs == [{"start": 0.1, "end": 0.4, "text": "はい。"}, {"start": 1.1, "end": 1.25, "text": "はい。"}]
    assert model.kwargs[0]["timestamps"] is True and model.kwargs[0]["verbose"] is False


def test_cli_rejects_zero_threads(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "--threads", "0"])
    with pytest.raises(SystemExit):
        cli.main()


def test_transcribe_passes_nemo_logs_on_to_load(stub_haas, restore_process_state):
    nemo_logging.setLevel(nemo_logging.INFO)
    t.transcribe("x.wav", model="m", nemo_logs=True)
    assert nemo_logging.getEffectiveLevel() == nemo_logging.INFO


def test_load_installs_and_quiets_nemo_before_loading_the_model(monkeypatch, restore_process_state):
    import haas_asr.prednet as prednet

    from haas_asr_cpu.kda import KimiDeltaAttentionCPU

    monkeypatch.setattr(prednet, "load_kda_class", lambda: "fla")
    seen = []
    monkeypatch.setattr(haas_asr, "load_model", lambda name, device, dtype=None: seen.append(
        (prednet.load_kda_class(), nemo_logging.getEffectiveLevel())) or _Model())
    nemo_logging.setLevel(nemo_logging.INFO)
    t.load("m")
    # NeMo logs the restore at INFO to stdout, so it must already be quiet while the model loads
    assert seen == [(KimiDeltaAttentionCPU, nemo_logging.ERROR)]


def test_fallback_message_goes_to_stderr_only(monkeypatch, stub_haas, capsys, restore_process_state):
    class Mismatch(_Model):
        def transcribe(self, chunks, batch_size, **kwargs):
            return [_Hyp("いいえ", [("はい", 0.1, 0.4)]) for _ in chunks]

    segs = t.transcribe("x.wav", model=Mismatch(), verbose=True)
    out, err = capsys.readouterr()
    assert segs == [{"start": 0.0, "end": 1.0, "text": "いいえ"}]
    assert "fell back" in err and out == ""
    t.transcribe("x.wav", model=Mismatch())  # verbose=False: silent
    assert capsys.readouterr() == ("", "")


def test_no_fallback_message_when_every_chunk_has_token_times(monkeypatch, stub_haas, capsys, restore_process_state):
    class Good(_Model):
        def transcribe(self, chunks, batch_size, **kwargs):
            return [_Hyp("はい", [("はい", 0.1, 0.4)]) for _ in chunks]

    t.transcribe("x.wav", model=Good(), verbose=True)
    err = capsys.readouterr().err
    assert "segments" in err and "fell back" not in err


def test_default_model_name(stub_haas, restore_process_state):
    assert t.DEFAULT_MODEL == "third-intelligence/Haas-ja-v1.0"
    t.load()
    t.transcribe("x.wav")
    assert stub_haas == [(t.DEFAULT_MODEL, "cpu")] * 2


def test_each_chunk_gets_its_own_audio_and_offset(monkeypatch, stub_haas, restore_process_state):
    audio = [0.0] * 16000 + [1.0] * 4001
    monkeypatch.setattr(haas_asr, "load_audio", lambda *_: audio)
    monkeypatch.setattr(haas_asr, "split_at_quiet_points", lambda a: [(0, 16000), (16000, 20001)])

    class PerChunk(_Model):
        def transcribe(self, chunks, batch_size, **kwargs):
            self.calls.append([(len(c), c[0]) for c in chunks])
            return [_Hyp("はい。" if c[0] == 0.0 else "いいえ。", [("はい。" if c[0] == 0.0 else "いいえ。", 0.1, 0.4)])
                    for c in chunks]

    model = PerChunk()
    segs = t.transcribe("x.wav", model=model)
    assert model.calls == [[(16000, 0.0), (4001, 1.0)]]
    # chunk 2: 1.0-1.2500625 s; its token 0.1-0.4 s becomes 1.1 s to the chunk end, rounded to ms
    assert segs == [{"start": 0.1, "end": 0.4, "text": "はい。"}, {"start": 1.1, "end": 1.25, "text": "いいえ。"}]


def test_fallback_segment_times_are_rounded():
    segs, ok = t.chunk_segments("いいえ", {"char": [{"char": ["はい"], "start": 0.1, "end": 0.4}]},
                                1.000625, 2.0000625)
    assert not ok and segs == [{"start": 1.001, "end": 2.0, "text": "いいえ"}]


def test_token_times_use_the_chunk_offset_and_end_to_the_millisecond():
    segs, ok = t.chunk_segments(" はい。 ", {"char": [{"char": ["はい。"], "start": 0.1, "end": 29.5}]},
                                1.000625, 30.0123)
    assert ok and segs == [{"start": 1.101, "end": 30.012, "text": "はい。"}]
    segs, ok = t.chunk_segments(" いいえ ", {"char": []}, 1.0, 2.0)
    assert not ok and segs[0]["text"] == "いいえ"


def test_model_load_time_is_logged(stub_haas, capsys, restore_process_state):
    t.transcribe("x.wav", verbose=True)
    assert "model loaded in" in capsys.readouterr().err


class _Cfg(dict):
    """Stand-in for the model's OmegaConf config: attribute access on a nested dict."""

    __getattr__ = dict.__getitem__


def _gpu_model():
    from omegaconf import OmegaConf

    m = _Model()
    m.cfg = _Cfg(decoding=OmegaConf.create({"greedy": {"use_cuda_graph_decoder": True}}))
    m.strategies = []
    m.change_decoding_strategy = lambda cfg, verbose=True: m.strategies.append(cfg.greedy.use_cuda_graph_decoder)
    return m


@pytest.fixture
def restore_gpu_state(monkeypatch):
    import nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base as llb

    monkeypatch.setattr(llb, "check_cuda_python_cuda_graphs_conditional_nodes_supported",
                        llb.check_cuda_python_cuda_graphs_conditional_nodes_supported)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", torch.backends.cudnn.enabled)
    return llb


def test_load_on_an_amd_gpu_applies_the_rocm_workarounds(monkeypatch, restore_process_state, restore_gpu_state):
    llb = restore_gpu_state
    original = llb.check_cuda_python_cuda_graphs_conditional_nodes_supported
    monkeypatch.setattr(torch.version, "hip", "7.16")
    for device in ("cuda", "cuda:0", torch.device("cuda", 1)):  # every way of naming a GPU
        model, seen = _gpu_model(), []
        monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
        monkeypatch.setattr(llb, "check_cuda_python_cuda_graphs_conditional_nodes_supported", original)
        monkeypatch.setattr(haas_asr, "load_model", lambda name, device, dtype=None: seen.append(
            (device, dtype, torch.backends.cudnn.enabled,
             llb.check_cuda_python_cuda_graphs_conditional_nodes_supported is original)) or model)
        assert t.load("m", device=device, dtype="bfloat16") is model
        # MIOpen off and NeMo's cuda-python check replaced before the model (and its decoder) is built
        assert seen == [(device, torch.bfloat16, False, False)]
        with pytest.raises(EnvironmentError, match="ROCm"):
            llb.check_cuda_python_cuda_graphs_conditional_nodes_supported()
        assert model.strategies == [False]  # no CUDA (HIP) graphs in the decoder


def test_rocm_pytorch_on_the_cpu_still_skips_the_nvidia_driver_check(monkeypatch, restore_process_state,
                                                                      restore_gpu_state):
    """NeMo checks NVIDIA's driver whenever PyTorch sees a GPU, even for a model on the CPU."""
    llb = restore_gpu_state
    original = llb.check_cuda_python_cuda_graphs_conditional_nodes_supported
    monkeypatch.setattr(torch.version, "hip", "7.16")
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    model, seen = _gpu_model(), []
    monkeypatch.setattr(haas_asr, "load_model", lambda name, device, dtype=None: seen.append(
        llb.check_cuda_python_cuda_graphs_conditional_nodes_supported is original) or model)
    t.load("m", device="cpu")
    assert seen == [False]  # replaced before the model (and its decoder) is built
    assert torch.backends.cudnn.enabled and model.strategies == []  # the GPU-only changes are not made


def test_load_on_cpu_or_an_nvidia_gpu_changes_nothing_else(monkeypatch, restore_process_state, restore_gpu_state):
    llb = restore_gpu_state
    check = llb.check_cuda_python_cuda_graphs_conditional_nodes_supported
    seen = []
    monkeypatch.setattr(haas_asr, "load_model", lambda name, device, dtype=None: seen.append((device, dtype)) or _Model())
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setattr(torch.version, "hip", None)
    t.load("m")
    t.load("m", device="cuda")  # NVIDIA: NeMo's own CUDA graphs and cuDNN are left alone
    assert seen == [("cpu", torch.float32), ("cuda", torch.float32)]
    assert torch.backends.cudnn.enabled and llb.check_cuda_python_cuda_graphs_conditional_nodes_supported is check


def test_unknown_dtype_is_rejected_before_loading(monkeypatch, restore_process_state):
    monkeypatch.setattr(haas_asr, "load_model", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(ValueError, match="dtype"):
        t.load("m", dtype="float16")


def test_transcribe_passes_device_and_dtype_to_load(monkeypatch, stub_haas, restore_process_state):
    seen = []
    monkeypatch.setattr(t, "load", lambda name, **k: seen.append((k["device"], k["dtype"])) or _Model())
    t.transcribe("x.wav", device="cuda:1", dtype="bfloat16")
    assert seen == [("cuda:1", "bfloat16")]


def test_chunks_cut_at_a_fraction_of_a_second(monkeypatch, stub_haas, restore_process_state):
    """Real cuts land on any sample: the second chunk starts at 1.5625 s, not a whole second."""
    monkeypatch.setattr(haas_asr, "load_audio", lambda *_: [0.0] * 48000)
    monkeypatch.setattr(haas_asr, "split_at_quiet_points", lambda a: [(0, 25000), (25000, 48000)])
    segs = t.transcribe("x.wav", model=_Model())
    assert segs == [{"start": 0.1, "end": 0.4, "text": "はい。"}, {"start": 1.663, "end": 1.962, "text": "はい。"}]


def test_every_sentence_of_every_chunk_is_returned(monkeypatch, stub_haas, restore_process_state):
    """A real ~30 s chunk holds several sentences; transcribe() must return all of them, in order."""
    monkeypatch.setattr(haas_asr, "load_audio", lambda *_: [0.0] * 32000)
    monkeypatch.setattr(haas_asr, "split_at_quiet_points", lambda a: [(0, 16000), (16000, 32000)])

    class TwoSentences(_Model):
        def transcribe(self, chunks, batch_size, **kwargs):
            return [_Hyp("はい。いいえ。", [("はい。", 0.1, 0.4), ("いいえ。", 0.5, 0.9)]) for _ in chunks]

    segs = t.transcribe("x.wav", model=TwoSentences())
    assert [(s["start"], s["end"], s["text"]) for s in segs] == [
        (0.1, 0.4, "はい。"), (0.5, 0.9, "いいえ。"), (1.1, 1.4, "はい。"), (1.5, 1.9, "いいえ。")]


def test_frame_zero_rule_applies_in_every_chunk_not_only_the_first():
    timestamp = {"char": [{"char": ["こ"], "start": 0.0, "end": 0.32}, {"char": ["ん"], "start": 16.4, "end": 16.56},
                          {"char": ["にちは。"], "start": 16.56, "end": 17.28}]}
    segs, ok = t.chunk_segments("こんにちは。", timestamp, 30.0, 60.0)  # the chunk from 30 s
    assert ok and segs == [{"start": 46.4, "end": 47.28, "text": "こんにちは。"}]
