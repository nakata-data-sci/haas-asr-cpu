"""haas-cpu-transcribe AUDIO [-o out.json] [--srt out.srt]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .transcribe import DEFAULT_MODEL, transcribe


def _ts(t: float, sep: str = ".") -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    return "".join(f"{i}\n{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}\n{s['text']}\n\n"
                   for i, s in enumerate(segments, 1))


def main() -> None:
    ap = argparse.ArgumentParser(prog="haas-cpu-transcribe", description="Transcribe Japanese audio with Haas-ja on CPU.")
    ap.add_argument("audio", type=Path)
    ap.add_argument("-o", "--json", type=Path, help="write segments as JSON")
    ap.add_argument("--srt", type=Path, help="write segments as SRT")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    # fail before the model is loaded and the (long) transcription runs, not after
    if not args.audio.is_file():
        ap.error(f"audio file not found: {args.audio}")
    if args.batch_size < 1 or (args.threads is not None and args.threads < 1):
        ap.error("--batch-size and --threads must be positive")
    for out in (args.json, args.srt):
        if out is None:
            continue
        if out.is_dir():
            ap.error(f"output path is a directory: {out}")
        if not out.parent.is_dir():  # not resolve(): "missing/../o.json" cannot be written either
            ap.error(f"directory does not exist: {out.resolve().parent}")
    if not sys.stdout.isatty():  # e.g. redirected on Japanese Windows (cp932 cannot encode every kanji)
        sys.stdout.reconfigure(encoding="utf-8")

    segs = transcribe(str(args.audio), model=args.model, threads=args.threads,
                      batch_size=args.batch_size, verbose=not args.quiet)
    for s in segs:  # print first, so the transcript is not lost if writing a file fails
        print(f"[{_ts(s['start'])} --> {_ts(s['end'])}] {s['text']}")
    sys.stdout.flush()
    if args.json:
        args.json.write_text(json.dumps({"model": args.model, "segments": segs}, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    if args.srt:
        args.srt.write_text(to_srt(segs), encoding="utf-8")


if __name__ == "__main__":
    main()
