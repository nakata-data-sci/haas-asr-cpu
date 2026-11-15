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
    for parent in (tmp_path / "nope" / "..", wav):  # a missing directory, and a regular file
        monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-o", str(parent / "o.json")])
        with pytest.raises(SystemExit):
            cli.main()
        assert str(parent) in capsys.readouterr().err  # names the directory that is missing
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
