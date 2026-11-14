from haas_asr_cpu.segments import build_segments, repair_tokens


def tok(c, s, e):
    return {"char": [c], "start": s, "end": e}


def test_splits_on_sentence_end_and_skips_blank():
    toks = [tok("", 0.0, 0.3), tok("みな", 11.7, 12.0), tok("さん", 12.0, 12.3), tok("?", 12.3, 12.4),
            tok("缶の", 13.0, 13.4), tok("裏側。", 13.4, 14.0)]
    assert build_segments(toks, offset=100.0) == [
        {"start": 111.7, "end": 112.4, "text": "みなさん?"},
        {"start": 113.0, "end": 114.0, "text": "缶の裏側。"},
    ]


def test_splits_on_pause():
    toks = [tok("あい", 0.0, 0.5), tok("うえ", 0.6, 1.0), tok("おか", 2.0, 2.3)]
    assert [s["text"] for s in build_segments(toks)] == ["あいうえ", "おか"]


def test_short_fragment_joins_next():
    toks = [tok("私", 29.9, 30.1), tok("はこの", 33.4, 33.8), tok("缶。", 33.8, 34.2)]
    assert build_segments(toks) == [{"start": 29.9, "end": 34.2, "text": "私はこの缶。"}]


def test_long_utterance_splits_at_short_pause_after_soft_max():
    toks = [tok("あ", i * 0.5, i * 0.5 + 0.5) for i in range(20)]  # 0-10 s, no pause
    toks += [tok("い", 10.4 + i * 0.5, 10.9 + i * 0.5) for i in range(4)]  # 0.4 s pause
    segs = build_segments(toks)
    assert [s["end"] for s in segs] == [10.0, 12.4]


def test_hard_max_without_any_break():
    toks = [tok("あ", i * 0.5, i * 0.5 + 0.5) for i in range(100)]
    assert all(s["end"] - s["start"] <= 20 for s in build_segments(toks))


def test_char_as_plain_string_and_missing_start():
    toks = [{"char": "あい", "start": 0.0, "end": 0.2}, {"char": ["う"], "start": None, "end": None}]
    assert build_segments(toks) == [{"start": 0.0, "end": 0.2, "text": "あい"}]


def test_repair_byte_fallback_tokens():
    """兎 is three byte-fallback tokens (each U+FFFD); restore it from the text."""
    toks = [tok("", 0.0, 0.1), tok("�", 0.16, 0.24), tok("�", 0.24, 0.32), tok("�", 0.32, 0.4),
            tok("に", 0.4, 0.56), tok("角。", 0.56, 0.8)]
    fixed = repair_tokens(toks, "兎に角。")
    assert [t["char"] for t in fixed if t["char"]] == ["兎", "に", "角。"]
    assert fixed[1]["start"] == 0.16 and fixed[1]["end"] == 0.4
    assert build_segments(fixed) == [{"start": 0.16, "end": 0.8, "text": "兎に角。"}]


def test_repair_rejects_mismatch():
    assert repair_tokens([tok("あ", 0, 1)], "い") is None
    assert repair_tokens([tok("あ", 0, 1)], "あい") is None  # text left over


def test_first_token_at_frame_zero_moves_to_speech():
    toks = [tok("こ", 0.0, 0.32), tok("ん", 16.4, 16.56), tok("にちは", 16.56, 17.28), tok("私は", 17.6, 18.2)]
    assert build_segments(toks)[0] == {"start": 16.4, "end": 18.2, "text": "こんにちは私は"}
