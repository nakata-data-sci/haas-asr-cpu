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
    monkeypatch.setattr(haas_asr, "load_model", lambda name, device: seen.append(
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
