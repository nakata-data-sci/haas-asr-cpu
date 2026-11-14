"""haas-cpu-transcribe AUDIO [-o out.json] [--srt out.srt]"""

from __future__ import annotations

import argparse
import json
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

    segs = transcribe(str(args.audio), model=args.model, threads=args.threads,
                      batch_size=args.batch_size, verbose=not args.quiet)
    if args.json:
        args.json.write_text(json.dumps({"model": args.model, "segments": segs}, ensure_ascii=False, indent=2))
    if args.srt:
        args.srt.write_text(to_srt(segs))
    for s in segs:
        print(f"[{_ts(s['start'])} --> {_ts(s['end'])}] {s['text']}")


if __name__ == "__main__":
    main()
