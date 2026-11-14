"""Transcribe an audio file with Haas-ja on CPU and return sentence-level segments."""

from __future__ import annotations

import sys
import time

from .kda import install
from .segments import build_segments, repair_tokens

DEFAULT_MODEL = "third-intelligence/Haas-ja-v1.0"


def transcribe(audio_path: str, model: str = DEFAULT_MODEL, threads: int | None = None,
               batch_size: int = 4, verbose: bool = False) -> list[dict]:
    """Return [{"start": sec, "end": sec, "text": str}, ...] for the whole file.

    The audio is cut into chunks of up to 30 s at quiet points (Haas's own long-form splitter), each chunk
    is decoded with token timestamps, and the tokens are grouped into sentences (see segments.py).
    With verbose=False, NeMo's logger (process-wide) is set to ERROR, since NeMo logs to stdout.
    """
    import torch

    if threads:
        torch.set_num_threads(threads)
    install()
    if not verbose:
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
    with torch.inference_mode():
        hyps = asr.transcribe(chunks, batch_size=batch_size, timestamps=True, verbose=False)
    if isinstance(hyps, tuple):
        hyps = hyps[0]

    segments, coarse = [], 0
    for (s, e), h in zip(bounds, hyps):
        c_start, c_end = s / sr, e / sr
        text = h.text.strip()
        tokens = repair_tokens(((getattr(h, "timestamp", None) or {}).get("char")) or [], text)
        segs = build_segments(tokens, offset=c_start) if tokens else []
        # If the tokens do not add up to the text, do not trust their times: one segment per chunk.
        if segs and "".join(x["text"] for x in segs).replace(" ", "") == text.replace(" ", ""):
            segments.extend(segs)
        elif text:
            coarse += 1
            segments.append({"start": round(c_start, 3), "end": round(c_end, 3), "text": text})
    if coarse:
        log(f"{coarse} chunk(s) fell back to chunk-level timestamps")
    log(f"{len(chunks)} chunks, {len(segments)} segments, {time.monotonic() - t0:.1f} s")
    return segments
