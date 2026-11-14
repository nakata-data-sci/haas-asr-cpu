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


def test_short_fragment_joins_next_without_pause():
    toks = [tok("え。", 0.5, 0.7), tok("次の", 0.9, 1.2), tok("話。", 1.2, 1.6)]
    assert build_segments(toks) == [{"start": 0.5, "end": 1.6, "text": "え。次の話。"}]


def test_short_fragment_stands_alone_across_pause():
    toks = [tok("私", 29.9, 30.1), tok("はこの", 33.4, 33.8), tok("缶。", 33.8, 34.2)]
    assert build_segments(toks) == [{"start": 29.9, "end": 30.1, "text": "私"},
                                    {"start": 33.4, "end": 34.2, "text": "はこの缶。"}]
    toks = [tok("あ", 0.5, 0.6), tok("今日は", 25.0, 25.5), tok("晴れ。", 25.5, 26.0)]
    assert [s["text"] for s in build_segments(toks)] == ["あ", "今日は晴れ。"]


def test_hard_max_checked_before_adding_a_token():
    toks = [tok("あ", 0.0, 19.5), tok("い", 19.6, 25.0)]
    assert [(s["start"], s["end"]) for s in build_segments(toks)] == [(0.0, 19.5), (19.6, 25.0)]


def test_long_utterance_splits_at_short_pause_after_soft_max():
    toks = [tok("あ", i * 0.5, i * 0.5 + 0.5) for i in range(20)]  # 0-10 s, no pause
    toks += [tok("い", 10.4 + i * 0.5, 10.9 + i * 0.5) for i in range(4)]  # 0.4 s pause
    segs = build_segments(toks)
    assert [s["end"] for s in segs] == [10.0, 12.4]


def test_hard_max_without_any_break():
    toks = [tok("あ", i * 0.5, i * 0.5 + 0.5) for i in range(100)]
    assert all(s["end"] - s["start"] <= 20 for s in build_segments(toks))


def test_char_as_plain_string_and_missing_times_keep_text():
    toks = [{"char": "あい", "start": 0.0, "end": 0.2}, {"char": ["う"], "start": None, "end": None}]
    assert build_segments(toks) == [{"start": 0.0, "end": 0.2, "text": "あいう"}]
    toks = [{"char": ["あ"], "start": None, "end": None}, tok("いう。", 1.0, 1.4), {"char": "え", "start": 1.5, "end": None}]
    assert "".join(s["text"] for s in build_segments(toks)) == "あいう。え"
    assert build_segments([{"char": ["あ"], "start": None, "end": None}]) == []


def test_repair_skips_full_width_space():
    assert [t["char"] for t in repair_tokens([tok("あ", 0, 1), tok("い", 1, 2)], "あ\u3000い")] == ["あ", "\u3000い"]


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


def test_full_width_marks_close_segments():
    toks = [tok("本当", 0.0, 0.4), tok("？", 0.4, 0.5), tok("はい", 0.6, 0.9), tok("！", 0.9, 1.0)]
    assert [s["text"] for s in build_segments(toks)] == ["本当？", "はい！"]
    long = [tok("あ", i * 0.5, i * 0.5 + 0.5) for i in range(17)] + [tok("，", 8.5, 8.6), tok("いい", 8.6, 9.0)]
    assert [s["text"][-1] for s in build_segments(long)] == ["，", "い"]


def test_repair_keeps_spaces_between_words():
    toks = [tok("TEDx", 0.0, 0.3), tok("Kyoto", 0.3, 0.6), tok("です。", 0.6, 0.9)]
    fixed = repair_tokens(toks, "TEDx Kyotoです。")
    assert build_segments(fixed) == [{"start": 0.0, "end": 0.9, "text": "TEDx Kyotoです。"}]


def test_repair_byte_fallback_followed_by_blank_token():
    toks = [tok("�", 0.1, 0.2), tok("�", 0.2, 0.3), tok("�", 0.3, 0.4), tok("", 0.4, 0.4), tok("に", 0.4, 0.5)]
    fixed = repair_tokens(toks, "兎に")
    assert [t["char"] for t in fixed if t["char"]] == ["兎", "に"]


def test_first_token_closing_a_sentence_stays():
    toks = [tok("はい。", 0.0, 0.3), tok("今日は", 2.0, 2.5), tok("晴れ。", 2.5, 3.0)]
    assert build_segments(toks)[0] == {"start": 0.0, "end": 0.3, "text": "はい。"}


def test_repair_two_byte_fallback_runs_with_blank_between():
    toks = [tok("�", 0.1, 0.2), tok("�", 0.2, 0.3), tok("�", 0.3, 0.4), tok("", 0.4, 0.4),
            tok("�", 0.5, 0.6), tok("�", 0.6, 0.7), tok("�", 0.7, 0.8), tok("に", 0.8, 0.9)]
    fixed = repair_tokens(toks, "兎鹿に")
    assert "".join(t["char"] for t in fixed) == "兎鹿に"
    assert build_segments(fixed) == [{"start": 0.1, "end": 0.9, "text": "兎鹿に"}]


def test_token_with_end_but_no_start_keeps_its_text_and_time():
    assert build_segments([{"char": ["あいう"], "start": None, "end": 2.2}]) == [
        {"start": 2.2, "end": 2.2, "text": "あいう"}]
    toks = [{"char": ["あの"], "start": None, "end": 0.5}, tok("今日は晴れ。", 1.0, 1.8)]
    segs = build_segments(toks)
    assert segs[0]["start"] == 0.5 and "".join(s["text"] for s in segs) == "あの今日は晴れ。"
