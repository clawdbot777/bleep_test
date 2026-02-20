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

## Deployment

### Docker (recommended)

The easiest way to run Bleeper — all dependencies (ffmpeg, CUDA, whisperX) are baked into the image.

**Requirements on the host:**
- Docker + docker compose
- NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU acceleration (optional — falls back to CPU)
- Unraid: install the **nvidia-driver** plugin via Community Applications

```bash
# 1. Clone the repo
git clone https://github.com/jakezp/bleep_test.git
cd bleep_test

# 2. Configure
cp .env.example .env
# edit .env – fill in PLEX_URL, PLEX_TOKEN, PLEX_SECTION_ID

# 3. Build and start
docker compose up -d --build

# 4. Check it's running
curl http://localhost:5000/api/list_files
```

**Key volume mounts** (edit `docker-compose.yml` to match your paths):

| Container path | Purpose |
|---|---|
| `/app/uploads` | Staging area – bleeper reads/writes temp files here |
| `/media` | Your media library – must match the paths Radarr/Sonarr/Plex use |

> **Unraid note:** set media volume to `/mnt/user/media:/media` (or wherever your share lives).

#### No GPU? CPU-only mode

Comment out the `deploy.resources` block in `docker-compose.yml` and set:
```yaml
environment:
  WHISPERX_COMPUTE_TYPE: int8
```
Transcription will be slower but fully functional.

---

### Manual Installation

```bash
pip install -r requirements.txt
# Also requires: ffmpeg + ffprobe in PATH, Python 3.11+
```

### Running (manual)

```bash
# Development
python run.py

# Production (single worker – required for in-process job tracking)
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 600 run:app
```

---

## Arr Stack Integration

### Radarr / Sonarr

1. **Settings → Connect → Custom Script**
2. Script path: `/opt/bleeper/arr_hook.py`
3. Events: ✅ On Import, ✅ On Upgrade

```bash
# Configure via environment variables (or edit the CONFIG block in arr_hook.py):
export BLEEPER_URL="http://bleeper:5000"      # use container name if on same Docker network
export BLEEPER_UPLOAD="/app/uploads"           # must match the container's upload volume
export PLEX_URL="http://plex:32400"
export PLEX_TOKEN="your_plex_token"
export PLEX_SECTION_ID="1"
```

> **Tip:** If Radarr/Sonarr run in Docker on the same `arr_net` network as bleeper, use `http://bleeper:5000` as the URL — no IP needed.

> **Unraid:** add arr_hook.py as a Custom Script under Settings → Connect in each *arr app.  The script is self-contained (just needs `requests` installed on the host, or run it inside a container that shares the network).

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
