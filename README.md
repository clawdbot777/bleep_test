# Bleeper – Automated Profanity Filter Pipeline

Automatically removes profanity from movies and TV shows: mutes/bleeps the audio, redacts subtitles, and produces a clean "Family" audio track alongside the original – all packaged back into the original container.

## Architecture

```
Sonarr/Radarr
     │ On Import event
     ▼
arr_hook.py  ──────────────────────┐
                                   │
watcher.py (drop folder)  ─────────┤
                                   ▼
                          POST /api/process_full
                                   │
                    ┌──────────────▼──────────────┐
                    │       Bleeper Flask API       │
                    │                               │
                    │  1. analyze_and_select_audio  │
                    │  2. normalize_audio (→ AC3 5.1│
                    │  3. extract_audio + FC channel│
                    │  4. whisperX transcribe       │
                    │  5. redact audio + subtitles  │
                    │  6. combine into MKV          │
                    │  7. cleanup temp files        │
                    └──────────────┬──────────────┘
                                   │
                          Plex library refresh
                                   │
                    ✅ Clean MKV in media library
```

## Pipeline Stages

| Stage | Endpoint | Description |
|-------|----------|-------------|
| 1 | `/api/analyze_and_select_audio` | Probe file; pick best audio stream (AC3 > DTS > TrueHD > EAC3, prefers English multichannel) |
| 2 | `/api/normalize_audio` | **Optional** – transcode any codec → AC3 5.1 to eliminate edge-cases downstream |
| 3 | `/api/extract_audio` | Extract selected stream + isolate center (FC) channel |
| 4 | `/api/transcribe` | whisperX STT → word-level timestamps (JSON + SRT) |
| 5 | `/api/redact` | Mute profanity + insert 800Hz bleep; redact subtitle text |
| 6 | `/api/combine_media` | Rebuild MKV: Family audio + Original audio + Redacted subs |
| 7 | `/api/cleanup` | Remove temp files; rename output to original filename |

### Fire-and-forget

```
POST /api/process_full
{
  "filename": "movie.mkv",
  "plex_url": "http://plex:32400",
  "plex_token": "YOUR_TOKEN",
  "plex_section_id": "1"
}
```

Returns immediately with a `job_id`.  Poll `/api/job_status/<job_id>` for progress.

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.11+
- ffmpeg + ffprobe (in PATH)
- whisperX (`pip install whisperx`)
- CUDA-capable GPU recommended for whisperX transcription

## Running

```bash
# Development
python run.py

# Production (single worker – required for in-process job tracking)
gunicorn -w 1 -b 0.0.0.0:5000 run:app
```

## Arr Stack Integration

### Radarr / Sonarr

1. **Settings → Connect → Custom Script**
2. Script path: `/opt/bleeper/arr_hook.py`
3. Events: ✅ On Import, ✅ On Upgrade

```bash
# Configure in arr_hook.py or via environment variables:
export BLEEPER_URL="http://bleeper:5000"
export BLEEPER_UPLOAD="/tmp/uploads"
export PLEX_URL="http://plex:32400"
export PLEX_TOKEN="your_plex_token"
export PLEX_SECTION_ID="1"
```

### Manual / Drop Folder

```bash
# Watch a folder and auto-submit new videos:
export WATCH_FOLDER="/media/incoming"
python watcher.py

# Or submit a single file manually:
python arr_hook.py --file /path/to/movie.mkv
```

## Configuration

Key settings in `app/bleeper_backend.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NORMALIZE_TARGET_CODEC` | `"ac3"` | Target codec for normalization. Set to `None` to skip. |
| `NORMALIZE_TARGET_CHANNELS` | `6` | Target channel count (6 = 5.1) |
| `NORMALIZE_TARGET_BITRATE` | `"640k"` | Target bitrate |
| `FILTER_LIST_PATH` | `app/filter_list.txt` | Path to profanity word list |

## API Reference

### `POST /api/initialize_job`
Create a new job. Returns `{"job_id": "..."}`.

### `POST /api/select_remote_file`
Associate a file already in the upload folder with a job.
```json
{"job_id": "...", "filename": "movie.mkv"}
```

### `POST /api/upload`
Upload a file (multipart/form-data). Fields: `file`, `job_id`.

### `POST /api/analyze_and_select_audio`
Probe the input and select the best audio stream.
```json
{"job_id": "..."}
```

### `POST /api/normalize_audio`
Transcode selected stream to normalized codec/layout.
```json
{"job_id": "...", "force": false}
```

### `POST /api/extract_audio`
Extract audio stream and center (FC) channel.
```json
{"job_id": "..."}
```

### `POST /api/transcribe`
Run whisperX on the center channel.
```json
{
  "job_id": "...",
  "whisperx_settings": {
    "model": "large-v3",
    "language": "en",
    "batch_size": 20,
    "compute_type": "float16"
  }
}
```

### `POST /api/redact`
Mute profanity + redact subtitles.
```json
{"job_id": "..."}
```

### `POST /api/combine_media`
Rebuild the final MKV.
```json
{"job_id": "..."}
```

### `POST /api/cleanup`
Remove temp files and rename output.
```json
{"job_id": "..."}
```

### `POST /api/process_full`
Run the entire pipeline in one call.
```json
{
  "filename": "movie.mkv",
  "whisperx_settings": {...},
  "plex_url": "http://plex:32400",
  "plex_token": "...",
  "plex_section_id": "1"
}
```
Returns HTTP 202 immediately:
```json
{"status": "queued", "job_id": "...", "message": "Pipeline started. Poll /api/job_status/<job_id> for progress."}
```

### `GET /api/job_status/<job_id>`
Poll pipeline progress.
```json
{"job_id": "...", "status": "running", "stage": "transcribe", "error": ""}
```

Status values: `initialized` → `queued` → `running` → `completed` | `failed`

### `GET /api/list_files`
List video files in the upload folder.

## Output

The final MKV contains:
- **Video**: original, untouched (stream copy)
- **Audio Track 1** (default): "Family audio" – profanity muted/bleeped, normalized
- **Audio Track 2**: "Original audio" – untouched original
- **Subtitle Track** (default): Redacted subtitles (profanity replaced with `***`)
