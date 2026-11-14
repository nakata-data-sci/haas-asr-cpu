from haas_asr_cpu.transcribe import chunk_segments


def tok(c, s, e):
    return {"char": [c], "start": s, "end": e}


def test_full_width_space_at_a_sentence_boundary_keeps_token_times():
    ts = {"char": [tok("今日は晴れです。", 0.0, 1.0), tok("明日は雨です。", 1.2, 2.0)]}
    segs, ok = chunk_segments("今日は晴れです。　明日は雨です。", ts, 10.0, 40.0)
    assert ok and [(s["start"], s["text"]) for s in segs] == [(10.0, "今日は晴れです。"), (11.2, "明日は雨です。")]


def test_times_are_clamped_to_the_chunk():
    segs, ok = chunk_segments("はい。", {"char": [tok("はい。", 29.8, 30.4)]}, 0.0, 30.0)
    assert ok and segs == [{"start": 29.8, "end": 30.0, "text": "はい。"}]


def test_mismatch_falls_back_to_one_chunk_segment():
    segs, ok = chunk_segments("いいえ", {"char": [tok("はい", 0.0, 0.5)]}, 5.0, 35.0)
    assert not ok and segs == [{"start": 5.0, "end": 35.0, "text": "いいえ"}]
    assert chunk_segments("あ", None, 0.0, 1.0) == ([{"start": 0.0, "end": 1.0, "text": "あ"}], False)


def test_empty_text():
    assert chunk_segments("  ", {"char": []}, 0.0, 1.0) == ([], True)


def test_model_given_as_path_is_loaded(monkeypatch):
    import importlib
    from pathlib import Path

    import pytest

    t = importlib.import_module("haas_asr_cpu.transcribe")  # the package attribute is the function
    seen = []

    def fake_load(m, **_):
        seen.append(m)
        raise RuntimeError("stop")

    import haas_asr

    monkeypatch.setattr(haas_asr, "load_audio", lambda *_: [0.0] * 16000)
    monkeypatch.setattr(t, "load", fake_load)
    with pytest.raises(RuntimeError):
        t.transcribe("x.wav", model=Path("m.nemo"))
    assert seen == ["m.nemo"]


def test_bad_audio_fails_before_the_model_is_loaded(monkeypatch, tmp_path):
    import importlib

    import pytest

    t = importlib.import_module("haas_asr_cpu.transcribe")
    monkeypatch.setattr(t, "load", lambda *a, **k: pytest.fail("model loaded before reading the audio"))
    bad = tmp_path / "empty.wav"
    bad.write_bytes(b"")
    with pytest.raises(Exception):
        t.transcribe(str(bad))
