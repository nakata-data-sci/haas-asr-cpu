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
    segs = [{"start": 0.0, "end": 1.5, "text": "兎𠮷。"}]
    monkeypatch.setattr(cli, "transcribe", lambda *a, **k: segs)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["haas-cpu-transcribe", str(wav), "-o", str(tmp_path / "o.json"),
                                      "--srt", str(tmp_path / "o.srt")])
    cli.main()
    assert "兎𠮷。" in capsys.readouterr().out
    assert (tmp_path / "o.srt").read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,500\n兎𠮷。\n\n"
    assert '"兎𠮷。"' in (tmp_path / "o.json").read_text(encoding="utf-8")
