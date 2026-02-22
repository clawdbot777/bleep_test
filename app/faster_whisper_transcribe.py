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
        [--batch_size 16] \
        [--max_segment_duration 6]

Accepts any audio format ffmpeg understands — no pre-conversion needed.
"""

import argparse
import json
import os
import sys

# Characters that mark a natural sentence end — split here when a segment is too long.
_SENTENCE_END = frozenset(".!?")


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


def _split_long_segments(segments: list[dict], max_duration: float) -> list[dict]:
    """
    Split segments that exceed max_duration seconds into shorter ones.

    Strategy:
      1. Walk the segment's words.
      2. Whenever we hit a sentence-ending word (ends with . ! ?) AND the
         current chunk has been running for >= 1 s, flush a new segment.
      3. If no sentence boundary exists within max_duration, force-split at
         the nearest word boundary after max_duration.

    Word-level timestamps are preserved; each child segment gets the exact
    start/end of its first/last word.
    """
    out = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        words    = seg.get("words", [])

        # Short enough or no word timestamps — keep as-is.
        if duration <= max_duration or not words:
            out.append(seg)
            continue

        chunk_words: list[dict] = []
        chunk_start: float      = words[0]["start"]

        for i, w in enumerate(words):
            chunk_words.append(w)
            chunk_duration = w["end"] - chunk_start
            is_last        = (i == len(words) - 1)
            at_sentence    = w["word"].rstrip().endswith(tuple(_SENTENCE_END))
            over_limit     = chunk_duration >= max_duration

            # Flush on: sentence end (if chunk > 1 s), forced split, or last word.
            if (at_sentence and chunk_duration >= 1.0) or over_limit or is_last:
                text = " ".join(cw["word"] for cw in chunk_words).strip()
                if text:
                    out.append({
                        "start": round(chunk_start, 3),
                        "end":   round(chunk_words[-1]["end"], 3),
                        "text":  text,
                        "words": chunk_words,
                    })
                chunk_words = []
                if not is_last:
                    chunk_start = words[i + 1]["start"]

    return out


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
    parser.add_argument("--beam_size",    type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=16,
                        help="Number of audio chunks processed in parallel.")
    parser.add_argument("--max_segment_duration", type=float, default=6.0,
                        help="Split segments longer than this many seconds (default 6).")
    parser.add_argument("--vad_filter",   action="store_true",
                        help="Use Silero VAD to skip non-speech audio (recommended for sparse-dialogue tracks).")
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
        print(f"[faster-whisper] Transcribing (sequential, vad={args.vad_filter}): {os.path.basename(args.audio)}", flush=True)
        segments_iter, info = model.transcribe(
            args.audio,
            word_timestamps=True,
            language=args.language,
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
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

    before = len(segments)
    segments = _split_long_segments(segments, args.max_segment_duration)
    after   = len(segments)
    if after > before:
        print(f"[faster-whisper] Segment split: {before} → {after} segments (max {args.max_segment_duration}s)", flush=True)

    whisperx_json = {"segments": segments}

    base      = os.path.splitext(os.path.basename(args.audio))[0]
    json_path = os.path.join(args.output_dir, f"{base}.json")
    srt_path  = os.path.join(args.output_dir, f"{base}.srt")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(whisperx_json, fh, indent=2, ensure_ascii=False)

    with open(srt_path, "w", encoding="utf-8") as fh:
        fh.write(_build_srt(segments))

    print(f"[faster-whisper] Done — {after} segments")
    print(f"[faster-whisper] JSON → {json_path}")
    print(f"[faster-whisper] SRT  → {srt_path}")


if __name__ == "__main__":
    main()
