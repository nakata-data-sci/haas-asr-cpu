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
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(tmp_path / "missing.wav")])
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

    assert json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))["segments"] == segs


def test_timestamps_over_a_minute_and_an_hour():
    assert cli._ts(3725.5) == "01:02:05.500" and cli._ts(59.9996) == "00:01:00.000"
    assert cli.to_srt([{"start": 61.25, "end": 3725.5, "text": "あ"}]) == "1\n00:01:01,250 --> 01:02:05,500\nあ\n\n"
