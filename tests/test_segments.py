from haas_asr_cpu.segments import build_segments, repair_tokens


def tok(c, s, e):
    return {"char": [c], "start": s, "end": e}


def test_splits_on_sentence_end_and_skips_blank():
    toks = [tok("", 11.4, 11.6), tok("みな", 11.7, 12.0), tok("さん", 12.0, 12.3), tok("?", 12.3, 12.4),
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
    segs = build_segments(toks)
    assert "".join(s["text"] for s in segs) == "あ" * 100 and len(segs) >= 3
    assert all(s["end"] - s["start"] <= 20 for s in segs)


def test_char_as_plain_string_and_missing_times_keep_text():
    toks = [{"char": "あい", "start": 0.0, "end": 0.2}, {"char": ["う"], "start": None, "end": None}]
    assert build_segments(toks) == [{"start": 0.0, "end": 0.2, "text": "あいう"}]
    toks = [{"char": ["あ"], "start": None, "end": None}, tok("いう。", 1.0, 1.4), {"char": "え", "start": 1.5, "end": None}]
    segs = build_segments(toks)
    assert "".join(s["text"] for s in segs) == "あいう。え"
    assert segs[0]["start"] == 1.0  # a leading token without times takes the next start
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
    toks = [tok("本当", 0.0, 0.4), tok("？", 0.4, 0.5), tok("はい", 0.6, 0.9), tok("！", 0.9, 1.0),
            tok("では", 1.1, 1.4)]
    assert [s["text"] for s in build_segments(toks)] == ["本当？", "はい！", "では"]
    for comma in ("，", "、"):
        long = [tok("あ", i * 0.5, i * 0.5 + 0.5) for i in range(17)] + [tok(comma, 8.5, 8.6), tok("いい", 8.6, 9.0)]
        assert [s["text"][-1] for s in build_segments(long)] == [comma, "い"]


def test_repair_keeps_spaces_between_words():
    toks = [tok("TEDx", 0.0, 0.3), tok("Kyoto", 0.3, 0.6), tok("です。", 0.6, 0.9)]
    fixed = repair_tokens(toks, "TEDx Kyotoです。")
    assert build_segments(fixed) == [{"start": 0.0, "end": 0.9, "text": "TEDx Kyotoです。"}]
    # as the Haas tokenizer really splits it: "▁" pieces decode to "" (one before "K")
    pieces = ["", "T", "E", "D", "x", "", "K", "y", "o", "t", "o", "です", "。"]
    toks = [tok(c, i * 0.1, i * 0.1 + 0.1) for i, c in enumerate(pieces)]
    assert build_segments(repair_tokens(toks, "TEDx Kyotoです。"))[0]["text"] == "TEDx Kyotoです。"


def test_repair_byte_fallback_followed_by_blank_token():
    toks = [tok("�", 0.1, 0.2), tok("�", 0.2, 0.3), tok("�", 0.3, 0.4), tok("", 0.4, 0.45), tok("に", 0.5, 0.6)]
    fixed = repair_tokens(toks, "兎に")
    assert [t["char"] for t in fixed if t["char"]] == ["兎", "に"]
    assert fixed[0]["end"] == 0.4  # the end of the last U+FFFD token, not of the blank after it


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


def test_first_token_closing_a_clause_stays():
    toks = [tok("でも、", 0.0, 0.3), tok("今日は", 2.0, 2.5), tok("晴れ。", 2.5, 3.0)]
    assert build_segments(toks)[0]["start"] == 0.0


def test_blank_tokens_do_not_split_a_pause():
    toks = [tok("あ", 0.5, 0.6), tok("", 1.0, 1.4), tok("い", 1.5, 1.7)]  # 0.9 s pause between あ and い
    assert [s["text"] for s in build_segments(toks)] == ["あ", "い"]


def test_ascii_marks_the_model_emits_close_segments():
    toks = [tok("本当", 0.0, 0.4), tok("!", 0.4, 0.5), tok("では", 0.6, 0.9)]
    assert [s["text"] for s in build_segments(toks)] == ["本当!", "では"]
    long = [tok("あ", i * 0.5, i * 0.5 + 0.5) for i in range(17)] + [tok(",", 8.5, 8.6), tok("いい", 8.6, 9.0)]
    assert [s["text"][-1] for s in build_segments(long)] == [",", "い"]


def test_thresholds_hold_for_frame_times_anywhere_in_a_chunk():
    f = lambda n: n * 0.01 * 8  # NeMo: frame * window_stride * subsampling
    for n in range(4, 404):  # the first token does not start at 0.0 (the frame-0 heuristic is tested above)
        # a pause of exactly 10 frames = 0.8 s closes the segment wherever it falls
        toks = [tok("今日は", f(n - 3), f(n)), tok("晴れです", f(n + 10), f(n + 13))]
        assert [s["text"] for s in build_segments(toks)] == ["今日は", "晴れです"], n
        # 250 frames = 20 s reaches hard_max: the first segment is exactly 20 s, not cut a frame early
        long = [tok("あ", f(n + i), f(n + i + 1)) for i in range(251)]
        first = build_segments(long)[0]
        assert round(first["end"] - first["start"], 6) == 20.0, n
        # 100 frames = 8 s reaches soft_max: a following 0.32 s pause (> short_gap) closes the segment
        soft = [tok("あ", f(n + i), f(n + i + 1)) for i in range(100)] + [tok("いい", f(n + 104), f(n + 106))]
        assert len(build_segments(soft)) == 2, n
    for k in range(1, 400):
        # frame-0 heuristic: a first token at 0.0 followed by exactly 0.8 s moves to the next token
        toks = [tok("あの", 0.0, f(k)), tok("今日は", f(k + 10), f(k + 13))]
        assert build_segments(toks)[0]["start"] == round(f(k + 10), 3), k


def test_comma_before_soft_max_does_not_split():
    toks = [tok("今日は", 0.1, 0.5), tok("、", 0.5, 0.5), tok("晴れです。", 0.6, 1.2)]
    assert [s["text"] for s in build_segments(toks)] == ["今日は、晴れです。"]


def test_comma_and_short_gap_at_exact_thresholds():
    f = lambda n: n * 0.01 * 8
    for n in range(4, 404):
        # a comma token ending exactly 100 frames (8 s) after the segment start closes it
        toks = [tok("あ", f(n + i), f(n + i + 1)) for i in range(99)] + [tok("、", f(n + 99), f(n + 100)),
                                                                          tok("いい", f(n + 100), f(n + 102))]
        assert len(build_segments(toks)) == 2, n
    # after 8 s, a pause of exactly short_gap (0.3 s) closes the segment
    toks = [tok("あ", i * 0.5, i * 0.5 + 0.5) for i in range(17)] + [tok("いい", 8.8, 9.0)]
    assert len(build_segments(toks)) == 2


def test_repair_rejects_a_byte_fallback_run_with_no_text():
    assert repair_tokens([tok("あ", 0, 1), tok("�", 1, 2), tok("い", 2, 3)], "あい") is None


def test_single_token_at_frame_zero():
    assert build_segments([tok("はい", 0.0, 0.3)]) == [{"start": 0.0, "end": 0.3, "text": "はい"}]


def test_no_comma_or_short_pause_split_before_soft_max():
    f = lambda n: n * 0.01 * 8
    for n in range(4, 404):
        # 99 frames = 7.92 s: a comma or a 0.32 s pause does not close the segment yet
        comma = [tok("あ", f(n + i), f(n + i + 1)) for i in range(98)] + [tok("、", f(n + 98), f(n + 99)),
                                                                         tok("いい", f(n + 99), f(n + 101))]
        assert len(build_segments(comma)) == 1, n
        pause = [tok("あ", f(n + i), f(n + i + 1)) for i in range(99)] + [tok("いい", f(n + 103), f(n + 105))]
        assert len(build_segments(pause)) == 1, n


def test_frame_zero_token_stays_before_a_pause_shorter_than_gap():
    f = lambda n: n * 0.01 * 8
    toks = [tok("あの", 0.0, f(3)), tok("今日は", f(12), f(15))]  # 9 frames = 0.72 s < 0.8 s
    assert build_segments(toks)[0]["start"] == 0.0


def test_filled_in_start_is_never_after_the_end():
    toks = [tok("今日は晴れ。", 0.5, 1.0), {"char": ["明日は雨"], "start": None, "end": 0.6}]
    segs = build_segments(toks)
    assert segs[-1] == {"start": 1.0, "end": 1.0, "text": "明日は雨"}  # not before the previous end
    assert all(s["start"] <= s["end"] for s in segs)


def test_repair_when_the_next_token_text_also_appears_elsewhere():
    bad = [tok("�", 0.2, 0.3), tok("�", 0.3, 0.4), tok("�", 0.4, 0.5)]
    fixed = repair_tokens([tok("に", 0.0, 0.2), *bad, tok("に", 0.5, 0.6)], "に兎に")
    assert [t["char"] for t in fixed if t["char"]] == ["に", "兎", "に"]
    fixed = repair_tokens([*bad, tok("に", 0.5, 0.6), tok("に", 0.6, 0.7)], "兎にに")
    assert [t["char"] for t in fixed if t["char"]] == ["兎", "に", "に"]


def test_missing_start_follows_the_previous_token():
    toks = [tok("今日は", 0.1, 1.0), {"char": ["晴れ。"], "start": None, "end": 3.0}]
    assert build_segments(toks) == [{"start": 0.1, "end": 3.0, "text": "今日は晴れ。"}]


def test_segment_never_ends_before_it_starts():
    toks = [tok("今日は", 0.5, 1.0), {"char": ["晴れ"], "start": None, "end": 0.4}]
    assert build_segments(toks) == [{"start": 0.5, "end": 1.0, "text": "今日は晴れ"}]
    assert build_segments([tok("あい", 1.0, 0.5)]) == [{"start": 1.0, "end": 1.0, "text": "あい"}]


def test_build_segments_does_not_change_the_callers_tokens():
    toks = [{"char": ["あ"], "start": None, "end": None}, tok("いう。", 1.0, 1.4)]
    before = [dict(t) for t in toks]
    build_segments(toks)
    assert toks == before


def test_segments_do_not_overlap_when_token_times_do():
    # the missing start is filled as min(2.0, 0.6) = 0.6; the pause is measured from the segment's end (2.0)
    toks = [tok("今日は", 0.1, 2.0), {"char": ["い"], "start": None, "end": 0.6}, tok("天気。", 1.5, 1.8)]
    segs = build_segments(toks)
    assert segs == [{"start": 0.1, "end": 2.0, "text": "今日はい天気。"}]
    toks = [tok("あああ", 0.1, 9.0), tok("いい", 0.2, 0.3), tok("う", 9.4, 9.5)]
    segs = build_segments(toks)  # 8.9 s long and 0.4 s after the segment's end: closes before う
    assert [s["text"] for s in segs] == ["あああいい", "う"]
    assert all(a["end"] <= b["start"] for a, b in zip(segs, segs[1:]))


def test_segment_end_is_rounded_with_a_fractional_offset():
    assert build_segments([tok("はい。", 0.1, 0.4)], offset=1.000625) == [
        {"start": 1.101, "end": 1.401, "text": "はい。"}]


def test_comma_split_measures_length_to_the_segments_end():
    toks = [tok("あああ", 0.5, 9.5), tok("いい、", 1.0, 1.2), tok("うう", 1.5, 1.7)]
    assert build_segments(toks) == [{"start": 0.5, "end": 9.5, "text": "あああいい、"},  # 9.0 s long at the comma
                                    {"start": 9.5, "end": 9.5, "text": "うう"}]  # not inside the previous one


def test_segments_never_overlap_after_any_kind_of_split():
    cases = [[tok("あああ。", 0.1, 2.0), tok("いい。", 0.5, 0.8)],  # sentence end
             [tok("あ", 0.1, 20.2), tok("い", 5.0, 6.0)],  # hard_max
             [tok("今日は", 0.1, 2.0), {"char": ["い"], "start": None, "end": 0.6}, tok("天気。", 1.5, 1.8)]]
    for toks in cases:
        segs = build_segments(toks)
        assert all(s["start"] <= s["end"] for s in segs)
        assert all(a["end"] <= b["start"] for a, b in zip(segs, segs[1:])), segs


def test_long_one_token_fragment_still_joins_after_a_short_pause():
    assert [s["text"] for s in build_segments([tok("あ", 0.1, 9.1), tok("いう", 9.5, 9.7)])] == ["あいう"]


def test_overlap_clamp_uses_the_latest_segment():
    toks = [tok("ああ。", 0.1, 1.0), tok("いい。", 1.0, 2.0), tok("うう。", 0.5, 0.6)]
    assert build_segments(toks)[-1] == {"start": 2.0, "end": 2.0, "text": "うう。"}


def test_segment_end_is_the_latest_end_of_any_token():
    # the latest end (5.0) is neither the first nor the last token's: no 0.8 s pause before えお
    toks = [tok("あ", 0.1, 0.2), tok("い", 0.2, 5.0), tok("う", 0.3, 0.4), tok("えお", 1.3, 1.5)]
    assert build_segments(toks) == [{"start": 0.1, "end": 5.0, "text": "あいうえお"}]
    # length at the comma is measured to the latest end (9.0), so it splits after え、
    toks = [tok("ああ", 0.1, 9.0), tok("い", 0.1, 0.2), tok("う", 0.2, 0.3), tok("え、", 0.3, 0.4),
            tok("おお", 0.5, 0.6)]
    assert [s["text"] for s in build_segments(toks)] == ["ああいうえ、", "おお"]


def test_thresholds_can_be_changed():
    toks = [tok("今日は", 0.1, 1.0), tok("晴れ", 1.6, 2.0)]  # 0.6 s pause
    assert len(build_segments(toks)) == 1 and len(build_segments(toks, gap=0.5)) == 2
    toks = [tok("あああ", 0.1, 5.0), tok("いい", 5.2, 5.5)]  # 4.9 s, then a 0.2 s pause
    assert len(build_segments(toks)) == 1
    assert len(build_segments(toks, soft_max=4.0, short_gap=0.1)) == 2
    assert len(build_segments(toks, soft_max=4.0)) == 1  # 0.2 s < short_gap 0.3
    toks = [tok("ああ", 0.1, 5.0), tok("いい、", 5.0, 5.5), tok("うう", 5.5, 6.0)]
    assert [s["text"] for s in build_segments(toks, soft_max=5.0)] == ["ああいい、", "うう"]
    toks = [tok("ああ", 0.1, 3.0), tok("いい", 3.0, 6.0)]
    assert len(build_segments(toks)) == 1 and len(build_segments(toks, hard_max=5.0)) == 2
    toks = [tok("あいう。", 0.1, 0.5), tok("えお", 0.6, 0.9)]
    assert len(build_segments(toks)) == 2 and len(build_segments(toks, min_chars=4)) == 1
    toks = [tok("あ", 0.0, 0.2), tok("い。", 0.7, 0.9)]  # frame-0 rule: a 0.5 s pause after a token at 0.0 s
    assert build_segments(toks)[0]["start"] == 0.0
    assert build_segments(toks, gap=0.5) == [{"start": 0.7, "end": 0.9, "text": "あい。"}]


def test_repair_byte_fallback_run_at_the_end_of_the_text():
    """A chunk cut after a rare kanji: the U+FFFD run (and a blank token) is the last thing in the text."""
    toks = [tok("白い", 0.1, 0.5), tok("�", 0.6, 0.7), tok("�", 0.7, 0.8), tok("�", 0.8, 0.9),
            tok("", 0.9, 0.9)]
    fixed = repair_tokens(toks, "白い兎")
    assert [t["char"] for t in fixed if t["char"]] == ["白い", "兎"]
    assert build_segments(fixed) == [{"start": 0.1, "end": 0.9, "text": "白い兎"}]
    two = [tok("�", 0.6 + 0.05 * n, 0.65 + 0.05 * n) for n in range(6)]  # 兎鷲: two rare kanji, 6 tokens
    fixed = repair_tokens([tok("白い", 0.1, 0.5), *two], "白い兎鷲")
    assert [t["char"] for t in fixed if t["char"]] == ["白い", "兎鷲"]


def test_hard_max_splits_when_a_token_crosses_20_s_by_a_little():
    toks = [tok("ああ", 0.1, 19.5), tok("いい", 19.5, 20.3)]  # 20.2 s together, just over hard_max
    assert build_segments(toks) == [{"start": 0.1, "end": 19.5, "text": "ああ"},
                                    {"start": 19.5, "end": 20.3, "text": "いい"}]


def test_repair_rejects_text_before_the_first_token():
    assert repair_tokens([tok("い", 0, 1)], "あい") is None  # "あ" has no token


def test_hard_max_before_adding_a_token_counts_from_the_segment_start():
    toks = [tok("あ", i * 0.5, i * 0.5 + 0.5) for i in range(39)]  # 0-19.5 s, no break
    toks.append(tok("い", 19.5, 20.3))  # 20.3 s from the first token, 0.8 s from the last
    assert [(s["start"], s["end"]) for s in build_segments(toks)] == [(0.0, 19.5), (19.5, 20.3)]


def test_repair_byte_fallback_run_after_a_multi_character_token():
    """The text position (6) and the token index (1) of the U+FFFD run differ."""
    bad = "�"
    toks = [tok("今日は公園で", 0.0, 0.5), tok(bad, 0.5, 0.6), tok(bad, 0.6, 0.7), tok(bad, 0.7, 0.8),
            tok("を", 0.8, 0.9), tok("見た。", 0.9, 1.2)]
    fixed = repair_tokens(toks, "今日は公園で兎を見た。")
    assert [t["char"] for t in fixed] == ["今日は公園で", "兎", "を", "見た。"]
    assert fixed[1]["start"] == 0.5 and fixed[1]["end"] == 0.8
    toks = [tok("雨を見て", 0.0, 0.5), tok(bad, 0.5, 0.6), tok(bad, 0.6, 0.7), tok(bad, 0.7, 0.8),
            tok("を", 0.8, 0.9), tok("見た。", 0.9, 1.2)]  # the next token's "を" also comes before the run
    assert [t["char"] for t in repair_tokens(toks, "雨を見て兎を見た。")] == ["雨を見て", "兎", "を", "見た。"]


def test_frame_zero_rule_skips_the_leading_blank_token():
    """The tokenizer's first piece is often a blank "▁" at 0.0 s; the rule looks at the first token with text."""
    toks = [tok("", 0.0, 0.0), tok("こ", 0.0, 0.32), tok("ん", 16.4, 16.56), tok("にちは。", 16.56, 17.28)]
    assert build_segments(toks) == [{"start": 16.4, "end": 17.28, "text": "こんにちは。"}]
    assert build_segments([tok("", 0.0, 0.0), tok("はい", 0.0, 0.3)]) == [{"start": 0.0, "end": 0.3, "text": "はい"}]


def test_a_segment_of_exactly_hard_max_is_closed():
    """20.0 s exactly, then tokens that do not make it longer (zero length at its end): they start the next one."""
    toks = [tok("今日は", 1.0, 11.0), tok("晴れ", 11.0, 21.0), tok("で", 21.0, 21.0), tok("す。", 21.0, 21.3)]
    assert build_segments(toks) == [{"start": 1.0, "end": 21.0, "text": "今日は晴れ"},
                                    {"start": 21.0, "end": 21.3, "text": "です。"}]
