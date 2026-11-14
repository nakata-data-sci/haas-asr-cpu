"""Build sentence-level segments from NeMo's per-token timestamps (timestamp["char"]).

NeMo groups "words" (split on spaces) into segments at . ! ?, so Japanese text, which has no spaces,
comes back as one segment per audio chunk. Here tokens are added one by one and a segment is closed at:
  - a token ending with a sentence-final mark (。?!？！)
  - a pause of `gap` seconds or more before the next token
  - once the segment is longer than `soft_max`: a comma (、,，) or a pause of `short_gap` or more
  - `hard_max` seconds (a long utterance with no comma or pause)
Fragments of `min_chars` characters or fewer are merged into the next segment, unless a pause of `gap`
or more separates them. A segment is longer than `hard_max` only if a single token is.

Rare kanji (e.g. 兎) are split into several byte-fallback tokens by SentencePiece, and each one decodes to
U+FFFD. `repair_tokens` aligns the tokens with the hypothesis text (h.text) and restores those characters.
"""

from __future__ import annotations

SENTENCE_END = ("\u3002", "?", "!", "\uff1f", "\uff01")  # 。 ? ! ？ ！
CLAUSE_END = ("\u3001", ",", "\uff0c")  # 、 , ，
BAD = "�"


def _text(tok: dict) -> str:
    c = tok.get("char")
    if isinstance(c, (list, tuple)):
        return "".join(c)
    return c or ""


def repair_tokens(tokens: list[dict], text: str) -> list[dict] | None:
    """Align token texts with `text`, replacing each run of U+FFFD tokens with the matching characters.

    Returns None if the tokens do not match the text."""
    out, pos, i = [], 0, 0
    while i < len(tokens):
        t = _text(tokens[i])
        if BAD not in t:
            stripped = t.strip()
            p0 = pos  # spaces skipped here stay with this token, so "TEDx Kyoto" keeps its space
            while pos < len(text) and text[pos] in " \u3000" and not text.startswith(stripped, pos):
                pos += 1
            if stripped and not text.startswith(stripped, pos):
                return None
            pos += len(stripped)
            out.append({**tokens[i], "char": text[p0:pos] if stripped else ""})
            i += 1
            continue
        j = i  # tokens [i, j) contain U+FFFD
        while j < len(tokens) and BAD in _text(tokens[j]):
            j += 1
        k = j  # the next token with visible text bounds the replacement
        while k < len(tokens) and not _text(tokens[k]).strip():
            k += 1
        if k < len(tokens):
            end = text.find(_text(tokens[k]).strip(), pos)
        else:
            end = len(text)
        if end <= pos:
            return None
        repl = text[pos:end]
        out.append({**tokens[i], "char": repl, "end": tokens[j - 1].get("end")})
        pos = end
        i = j
    if text[pos:].strip():
        return None
    return out


def _fill_times(tokens: list[dict]) -> list[dict]:
    """Keep every token with text. A missing start or end is taken from the neighbouring tokens."""
    toks = [dict(t) for t in tokens if _text(t).strip()]
    prev = None
    for t in toks:  # forward: a missing start follows the previous token
        if t.get("start") is None and prev is not None:
            t["start"] = prev
        if t.get("end") is None and t.get("start") is not None:
            t["end"] = t["start"]
        if t.get("end") is not None:
            prev = t["end"]
    nxt = None
    for t in reversed(toks):  # backward: leading tokens without times take the next start
        if t.get("start") is None and nxt is not None:
            t["start"] = t["end"] = nxt
        if t.get("start") is not None:
            nxt = t["start"]
    return [t for t in toks if t.get("start") is not None]  # empty only if no token has a time


def build_segments(tokens: list[dict], offset: float = 0.0, gap: float = 0.8, short_gap: float = 0.3,
                   soft_max: float = 8.0, hard_max: float = 20.0, min_chars: int = 2) -> list[dict]:
    """tokens: [{"char": [str] | str, "start": sec, "end": sec}, ...] relative to the chunk; offset = chunk start."""
    out: list[dict] = []
    cur: list[tuple[str, float, float]] = []

    def flush(force: bool = False):
        text = "".join(t for t, _, _ in cur).strip()
        if not text:
            cur.clear()
            return
        if len(text) <= min_chars and not force:
            return  # keep a short fragment for the next segment
        out.append({"start": round(offset + cur[0][1], 3),
                    "end": round(offset + cur[-1][2], 3), "text": text})
        cur.clear()

    timed = _fill_times(tokens)
    # Heuristic: TDT sometimes emits the first token of a chunk at frame 0 although its sound comes much
    # later (seen: "こ" at 0.0 s, "んにちは" from 16.4 s). Such a token is moved to just before the next one.
    # A token that closes a sentence or clause on its own (e.g. "はい。") is left where it is. Real speech
    # at the very start of a chunk followed by a pause would be moved too; chunks are cut at quiet points,
    # so this is rare.
    if (len(timed) > 1 and float(timed[0]["start"]) == 0.0
            and float(timed[1]["start"]) - float(timed[0]["end"]) >= gap
            and not _text(timed[0]).strip().endswith(SENTENCE_END + CLAUSE_END)):
        timed[0] = {**timed[0], "start": timed[1]["start"], "end": timed[1]["start"]}

    for tok in timed:
        text = _text(tok)
        start, end = float(tok["start"]), float(tok["end"])
        if cur:
            pause = start - cur[-1][2]
            long_enough = cur[-1][2] - cur[0][1] >= soft_max
            if pause >= gap or end - cur[0][1] > hard_max:
                flush(force=True)  # a short fragment is not carried across a pause or past hard_max
            elif long_enough and pause >= short_gap:
                flush()
        cur.append((text, start, end))
        dur = end - cur[0][1]
        if text.endswith(SENTENCE_END) or (dur >= soft_max and text.endswith(CLAUSE_END)):
            flush()
        elif dur >= hard_max:
            flush(force=True)
    flush(force=True)
    return out
