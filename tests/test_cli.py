import sys

import pytest

from haas_asr_cpu import cli


def test_missing_output_directory_fails_before_transcribing(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(cli, "transcribe", lambda *a, **k: called.append(1) or [])
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    for argv in (["--json", str(tmp_path / "nope" / "o.json")], ["--srt", str(tmp_path)],
                 ["--batch-size", "0"], ["--threads", "-2"]):
        monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), *argv])
        with pytest.raises(SystemExit):
            cli.main()
    for audio in (tmp_path / "missing.wav", tmp_path):  # missing, and a directory
        monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(audio)])
        with pytest.raises(SystemExit):
            cli.main()
    assert not called


def test_writes_json_and_srt(monkeypatch, tmp_path, capsys):
    segs = [{"start": 0.0, "end": 1.5, "text": "兎𠮷。"}, {"start": 2.0, "end": 2.5, "text": "はい。"}]
    monkeypatch.setattr(cli, "transcribe", lambda *a, **k: segs)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-o", str(tmp_path / "o.json"),
                                      "--srt", str(tmp_path / "o.srt")])
    cli.main()
    assert "兎𠮷。" in capsys.readouterr().out
    assert (tmp_path / "o.srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,500\n兎𠮷。\n\n2\n00:00:02,000 --> 00:00:02,500\nはい。\n\n")
    import json

    raw = (tmp_path / "o.json").read_text(encoding="utf-8")
    assert "兎𠮷。" in raw  # readable, not \\u escapes
    assert json.loads(raw)["segments"] == segs and "model" in json.loads(raw)


def test_timestamps_over_a_minute_and_an_hour():
    assert cli._ts(3725.5) == "01:02:05.500" and cli._ts(59.9996) == "00:01:00.000"
    assert cli.to_srt([{"start": 61.25, "end": 3725.5, "text": "あ"}]) == "1\n00:01:01,250 --> 01:02:05,500\nあ\n\n"


def test_redirected_stdout_is_utf8_even_if_the_locale_is_cp932(monkeypatch, tmp_path):
    import io

    monkeypatch.setattr(cli, "transcribe", lambda *a, **k: [{"start": 0.0, "end": 1.0, "text": "𠮷野家。"}])
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="cp932")  # not a TTY, like a redirect on Japanese Windows
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q"])
    cli.main()
    out.flush()
    assert "𠮷野家。" in raw.getvalue().decode("utf-8")


def test_output_under_a_missing_directory_with_dotdot_is_rejected_early(monkeypatch, tmp_path, capsys):
    called = []
    monkeypatch.setattr(cli, "transcribe", lambda *a, **k: called.append(1) or [])
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    for flag in ("-o", "--srt"):
        for parent in (tmp_path / "nope" / "..", tmp_path / "nope", wav):  # missing dirs, and a regular file
            monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), flag, str(parent / "o.out")])
            with pytest.raises(SystemExit):
                cli.main()
            err = capsys.readouterr().err
            assert f"directory does not exist: {parent}\n" in err  # names the directory, not the file
    assert not called


def test_default_model_and_json_records_the_model_used(monkeypatch, tmp_path):
    import json

    seen = []
    monkeypatch.setattr(cli, "transcribe", lambda *a, **k: seen.append(k["model"]) or [])
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q"])
    cli.main()
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q", "--model", "my/model", "-o",
                                      str(tmp_path / "o.json")])
    cli.main()
    assert seen == ["third-intelligence/Haas-ja-v1.0", "my/model"]
    assert json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))["model"] == "my/model"


def test_one_thread_and_batch_size_one_are_accepted(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "transcribe", lambda *a, **k: seen.append((k["threads"], k["batch_size"])) or [])
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q", "--threads", "1", "--batch-size", "1"])
    cli.main()
    assert seen == [(1, 1)]


def test_package_api_as_in_the_readme():
    import haas_asr_cpu

    assert set(haas_asr_cpu.__all__) == {"DEFAULT_MODEL", "KimiDeltaAttentionCPU", "build_segments", "install",
                                         "load", "repair_tokens", "transcribe"}
    for name in haas_asr_cpu.__all__:
        obj = getattr(haas_asr_cpu, name)
        assert isinstance(obj, str) if name == "DEFAULT_MODEL" else callable(obj), name


def test_no_speech_still_writes_empty_output_files(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(cli, "transcribe", lambda *a, **k: [])
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q", "--json", str(tmp_path / "o.json"),
                                      "--srt", str(tmp_path / "o.srt")])
    cli.main()  # a later step in a pipeline can rely on the files being there
    assert (tmp_path / "o.srt").read_text(encoding="utf-8") == ""
    assert json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))["segments"] == []


def test_device_and_dtype_options(monkeypatch, tmp_path):
    import torch

    seen = []
    monkeypatch.setattr(cli, "transcribe", lambda *a, **k: seen.append((k["device"], k["dtype"])) or [])
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q"])
    cli.main()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q", "--device", "cuda", "--dtype", "bfloat16"])
    cli.main()
    assert seen == [("cpu", "float32"), ("cuda", "bfloat16")]
    for device in ("cuda:1", "mps", "gpu"):  # a second GPU that is not there, not CUDA/ROCm, not a device
        monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q", "--device", device])
        with pytest.raises(SystemExit):
            cli.main()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q", "--device", "cuda"])
    with pytest.raises(SystemExit):
        cli.main()
    assert len(seen) == 2  # rejected before transcribing


def test_cli_passes_the_audio_path_and_writes_files_as_utf8(monkeypatch, tmp_path):
    """Files are written as UTF-8 whatever the locale (cp932 on Japanese Windows)."""
    from pathlib import Path

    seen, encodings = [], []
    monkeypatch.setattr(cli, "transcribe", lambda audio, **k: seen.append(audio) or [
        {"start": 0.0, "end": 1.0, "text": "𠮷野家。"}])
    real = Path.write_text
    monkeypatch.setattr(Path, "write_text", lambda self, data, encoding=None, **k: encodings.append(encoding)
                        or real(self, data, encoding=encoding, **k))
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-q", "-o", str(tmp_path / "o.json"),
                                      "--srt", str(tmp_path / "o.srt")])
    cli.main()
    assert seen == [str(wav)] and encodings == ["utf-8", "utf-8"]
