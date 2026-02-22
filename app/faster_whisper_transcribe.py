#!/usr/bin/env python3
"""
faster-whisper transcription — outputs WhisperX-compatible JSON + SRT.

Usage:
    python faster_whisper_transcribe.py <audio_file> \
        --output_dir <dir> \
        [--model large-v3] \
        [--device cuda] \
        [--compute_type float16] \
        [--language en] \
        [--beam_size 5] \
        [--batch_size 8]

Accepts any audio format ffmpeg understands — no pre-conversion needed.
"""

import argparse
import json
import os
import sys


def _to_srt_time(s: float) -> str:
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sec = int(s % 60)
    ms  = int(round((s % 1) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _build_srt(segments: list[dict]) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        start = _to_srt_time(seg["start"])
        end   = _to_srt_time(seg["end"])
        text  = seg["text"].strip()
        if text:
            blocks.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with faster-whisper, write WhisperX-compatible JSON + SRT."
    )
    parser.add_argument("audio",          help="Path to the audio file.")
    parser.add_argument("--output_dir",   default=".", help="Directory for output files.")
    parser.add_argument("--model",        default="large-v3")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--compute_type", default="float16")
    parser.add_argument("--language",     default=None, help="Language code (e.g. en). None = auto-detect.")
    parser.add_argument("--beam_size",    type=int, default=5)
    parser.add_argument("--batch_size",   type=int, default=16,
                        help="Number of audio chunks processed in parallel.")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: faster-whisper not installed. Run: pip install faster-whisper",
              file=sys.stderr)
        sys.exit(2)

    print(f"[faster-whisper] Loading {args.model} on {args.device} ({args.compute_type})", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    if args.batch_size > 1:
        from faster_whisper.transcribe import BatchedInferencePipeline
        pipeline = BatchedInferencePipeline(model=model)
        print(f"[faster-whisper] Transcribing (batch_size={args.batch_size}): {os.path.basename(args.audio)}", flush=True)
        segments_iter, info = pipeline.transcribe(
            args.audio,
            word_timestamps=True,
            language=args.language,
            batch_size=args.batch_size,
        )
    else:
        print(f"[faster-whisper] Transcribing (sequential): {os.path.basename(args.audio)}", flush=True)
        segments_iter, info = model.transcribe(
            args.audio,
            word_timestamps=True,
            language=args.language,
            beam_size=args.beam_size,
        )

    print(f"[faster-whisper] Language: {info.language} ({info.language_probability:.0%})", flush=True)

    segments = []
    for seg in segments_iter:
        words = [
            {"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
            for w in (seg.words or [])
        ]
        segments.append({
            "start": round(seg.start, 3),
            "end":   round(seg.end, 3),
            "text":  seg.text,
            "words": words,
        })

    whisperx_json = {"segments": segments}

    base      = os.path.splitext(os.path.basename(args.audio))[0]
    json_path = os.path.join(args.output_dir, f"{base}.json")
    srt_path  = os.path.join(args.output_dir, f"{base}.srt")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(whisperx_json, fh, indent=2, ensure_ascii=False)

    with open(srt_path, "w", encoding="utf-8") as fh:
        fh.write(_build_srt(segments))

    print(f"[faster-whisper] Done — {len(segments)} segments")
    print(f"[faster-whisper] JSON → {json_path}")
    print(f"[faster-whisper] SRT  → {srt_path}")


if __name__ == "__main__":
    main()
