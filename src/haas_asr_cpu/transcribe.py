"""Transcribe an audio file with Haas-ja on CPU and return sentence-level segments."""

from __future__ import annotations

import re
import sys
import time

from .kda import install
from .segments import build_segments, repair_tokens

DEFAULT_MODEL = "third-intelligence/Haas-ja-v1.0"


def _squash(s: str) -> str:
    return re.sub(r"\s", "", s)  # all whitespace, including U+3000, which str.strip() also removes


def transcribe(audio_path: str, model: str = DEFAULT_MODEL, threads: int | None = None,
               batch_size: int = 4, verbose: bool = False, nemo_logs: bool = False) -> list[dict]:
    """Return [{"start": sec, "end": sec, "text": str}, ...] for the whole file.

    The audio is cut at quiet points into chunks of about 30 s, at most 32 s (Haas's own long-form splitter); each chunk
    is decoded with token timestamps, and the tokens are grouped into sentences (see segments.py).
    verbose prints progress to stderr. Unless nemo_logs=True, NeMo's logger (process-wide) is set to ERROR,
    since NeMo writes its INFO logs to stdout.
    """
    import torch

    if threads:
        torch.set_num_threads(threads)
    install()
    if not nemo_logs:
        from nemo.utils import logging as nemo_logging

        nemo_logging.setLevel(nemo_logging.ERROR)  # NeMo logs to stdout; keep it for the transcript
    import haas_asr

    t0 = time.monotonic()
    asr = haas_asr.load_model(model, device="cpu")  # float32 on CPU
    log = (lambda m: print(f"[haas-asr-cpu] {m}", file=sys.stderr)) if verbose else (lambda m: None)
    log(f"model loaded in {time.monotonic() - t0:.1f} s")

    audio = haas_asr.load_audio(str(audio_path))
    sr = haas_asr.SAMPLE_RATE
    bounds = haas_asr.split_at_quiet_points(audio)
    chunks = [audio[s:e] for s, e in bounds]
    if not chunks:  # empty audio
        return []
    with torch.inference_mode():
        hyps = asr.transcribe(chunks, batch_size=batch_size, timestamps=True, verbose=False)
    if isinstance(hyps, tuple):
        hyps = hyps[0]

    segments, coarse = [], 0
    for (s, e), h in zip(bounds, hyps):
        segs, ok = chunk_segments(h.text, getattr(h, "timestamp", None), s / sr, e / sr)
        segments.extend(segs)
        coarse += not ok
    if coarse:
        log(f"{coarse} chunk(s) fell back to chunk-level timestamps")
    log(f"{len(chunks)} chunks, {len(segments)} segments, {time.monotonic() - t0:.1f} s")
    return segments


def chunk_segments(text: str, timestamp, c_start: float, c_end: float) -> tuple[list[dict], bool]:
    """Segments of one chunk from its hypothesis text and NeMo timestamp dict.

    Returns (segments, True), or a single chunk-level segment and False when the token times cannot be
    trusted (the tokens do not add up to the text). An empty text gives ([], True).
    """
    text = text.strip()
    if not text:
        return [], True
    tokens = repair_tokens((timestamp.get("char") if isinstance(timestamp, dict) else None) or [], text)
    segs = build_segments(tokens, offset=c_start) if tokens else []
    if segs and _squash("".join(x["text"] for x in segs)) == _squash(text):
        for x in segs:  # a token's end (start + duration) may run past the chunk
            x["end"] = min(x["end"], round(c_end, 3))
            x["start"] = min(x["start"], x["end"])
        return segs, True
    return [{"start": round(c_start, 3), "end": round(c_end, 3), "text": text}], False
