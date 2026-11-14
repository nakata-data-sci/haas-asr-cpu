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
    monkeypatch.setattr(haas_asr, "load_model", lambda name, device: loaded.append((name, device)) or _Model())
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
    monkeypatch.setattr(haas_asr, "load_model", lambda name, device: stub_haas.append(name) or model)
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "--model", "my/model", "--threads", "2",
                                      "--batch-size", "3"])
    cli.main()
    out, err = capsys.readouterr()
    assert out == "[00:00:00.100 --> 00:00:00.400] はい。\n"
    assert "[haas-asr-cpu]" in err and "[haas-asr-cpu]" not in out
    assert stub_haas == ["my/model"] and model.calls == [3] and torch.get_num_threads() == 2
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q"])
    cli.main()
    assert "[haas-asr-cpu]" not in capsys.readouterr().err


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
