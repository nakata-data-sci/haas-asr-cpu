"""Build sentence-level segments from NeMo's per-token timestamps (timestamp["char"]).

NeMo groups "words" (split on spaces) into segments at . ! ?, so Japanese text, which has no spaces,
comes back as one segment per audio chunk. Here tokens are added one by one and a segment is closed at:
  - a token ending with a sentence-final mark (。?!？！)
  - a pause of `gap` seconds or more before the next token
  - once the segment is longer than `soft_max`: a comma (、,，) or a pause of `short_gap` or more
  - `hard_max` seconds (a long utterance with no comma or pause)
Fragments of `min_chars` characters or fewer are merged into the next segment.

Rare kanji (e.g. 兎) are split into several byte-fallback tokens by SentencePiece, and each one decodes to
U+FFFD. `repair_tokens` aligns the tokens with the hypothesis text (h.text) and restores those characters.
"""

from __future__ import annotations

SENTENCE_END = ("。", "?", "!", "?", "!")
CLAUSE_END = ("、", ",", ",")
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
            while pos < len(text) and text[pos] == " " and not text.startswith(stripped, pos):
                pos += 1
            if stripped and not text.startswith(stripped, pos):
                return None
            out.append({**tokens[i], "char": stripped})
            pos += len(stripped)
            i += 1
            continue
        j = i  # tokens [i, j) contain U+FFFD
        while j < len(tokens) and BAD in _text(tokens[j]):
            j += 1
        if j < len(tokens):
            nxt = _text(tokens[j]).strip()
            end = text.find(nxt, pos) if nxt else len(text)
        else:
            end = len(text)
        if end < pos:
            return None
        repl = text[pos:end]
        out.append({**tokens[i], "char": repl, "end": tokens[j - 1].get("end")})
        pos = end
        i = j
    if text[pos:].strip():
        return None
    return out


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

    timed = [t for t in tokens if _text(t).strip() and t.get("start") is not None]
    # TDT sometimes emits the first token of a chunk at frame 0 although its sound comes much later
    # (seen: "こ" at 0.0 s, "んにちは" from 16.4 s). Move such a token to just before the next one.
    if len(timed) > 1 and float(timed[0]["start"]) == 0.0 and float(timed[1]["start"]) - float(timed[0]["end"]) >= gap:
        timed[0] = {**timed[0], "start": timed[1]["start"], "end": timed[1]["start"]}

    for tok in timed:
        text = _text(tok)
        start, end = float(tok["start"]), float(tok["end"])
        if cur:
            pause = start - cur[-1][2]
            long_enough = cur[-1][2] - cur[0][1] >= soft_max
            if pause >= gap or (long_enough and pause >= short_gap):
                flush()
        cur.append((text, start, end))
        dur = end - cur[0][1]
        if text.endswith(SENTENCE_END) or (dur >= soft_max and text.endswith(CLAUSE_END)):
            flush()
        elif dur >= hard_max:
            flush(force=True)
    flush(force=True)
    return out
