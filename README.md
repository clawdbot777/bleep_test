# Bleeper – Automated Profanity Filter Pipeline

Automatically removes profanity from movies and TV shows: mutes/bleeps the audio, redacts subtitles, and produces a clean "Family" audio track alongside the original – all packaged back into the original container.

## Architecture

```
Sonarr/Radarr
     │ On Import event
     ▼
arr_hook.sh ───────────────────────┐
                                   │
manual_bleeper.sh (manual) ────────┤
watcher.py (drop folder)  ─────────┤
                                   ▼
                          POST /api/process_full
                                   │
                    ┌──────────────▼──────────────┐
                    │       Bleeper Flask API       │
                    │    (single-worker queue)      │
                    │                               │
                    │  1. analyze & select audio    │
                    │  2. normalize → AC3 5.1       │
                    │  3. extract FC channel        │
                    │  4. whisperX transcribe       │
                    │  5. redact audio + subs       │
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
| 2 | `/api/normalize_audio` | Transcode any codec → AC3 5.1 — eliminates codec edge-cases downstream |
| 3 | `/api/extract_audio` | Extract selected stream + isolate center (FC) channel |
| 4 | `/api/transcribe` | whisperX STT → word-level timestamps (JSON + SRT) |
| 5 | `/api/redact` | Mute profanity + insert 800Hz bleep; redact subtitle text |
| 6 | `/api/combine_media` | Rebuild MKV: Family audio (default) + Original audio + kept subs + Redacted SRT |
| 7 | `/api/cleanup` | Move output back to original media path; remove all temp files |

### Job queue

Jobs run **one at a time** — the pipeline uses a single-worker queue so the GPU is never shared between concurrent transcription jobs. Submit as many as you like; each waits its turn.

### Resume / retry

If a job fails mid-pipeline, resubmitting the same file resumes from the failed stage — completed stages are skipped. Cleanup deletes the config on success so the next run always starts fresh.

### Fire-and-forget

```
POST /api/process_full
{
  "filename": "/data/media/movies/Film (2024)/Film (2024).mkv",
  "plex_url": "http://plex:32400",
  "plex_token": "YOUR_TOKEN",
  "plex_section_id": "1,2"
}
```

Returns HTTP 202 immediately with a `job_id` and queue `position`. Poll `/api/job_status/<job_id>` for progress.

## Deployment

### Docker (recommended)

All dependencies (ffmpeg, CUDA, whisperX, Python) are baked into the image.

**Requirements:**
- Docker + docker compose
- NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU acceleration (optional — falls back to CPU)
- Unraid: install the **nvidia-driver** plugin via Community Applications

```bash
# 1. Clone
git clone https://github.com/jakezp/bleep_test.git
cd bleep_test

# 2. Configure
cp .env.example .env
# edit .env – fill in PLEX_URL, PLEX_TOKEN, PLEX_SECTION_ID

# 3. Start
docker compose up -d --build

# 4. Verify
curl http://localhost:5000/api/list_files
```

**Volume mounts** (edit `docker-compose.yml`):

| Host path | Container path | Purpose |
|---|---|---|
| `/mnt/user/appdata/bleeper/uploads` | `/app/uploads` | Staging area for temp files (use cache pool) |
| `/mnt/user/media` | `/data/media` | Media library — matches arr stack convention |

> The container uses `/data/media` to match the standard arr stack path convention. No path translation needed between Radarr/Sonarr and Bleeper.

#### No GPU? CPU-only mode

Remove the `deploy.resources` block in `docker-compose.yml` and set:
```yaml
environment:
  WHISPERX_COMPUTE_TYPE: int8
```

---

### Unraid template

One-command import:

```bash
mkdir -p /boot/config/plugins/dockerMan/templates-user
curl -o /boot/config/plugins/dockerMan/templates-user/bleeper.xml \
  https://raw.githubusercontent.com/jakezp/bleep_test/main/unraid/bleeper.xml
```

Then Docker tab → **Add Container** → select `bleeper` from the template dropdown.

---

## Arr Stack Integration

### Radarr / Sonarr

1. **Settings → Connect → Custom Script**
2. Script path: `/path/to/arr_hook.sh`
3. Events: ✅ On Import, ✅ On Upgrade

Edit the config block at the top of `arr_hook.sh`:

```sh
BLEEPER_URL="http://bleeper:5000"   # container name if on same Docker network
PLEX_URL="http://plex:32400"
PLEX_TOKEN="your_plex_token"
PLEX_SECTION_ID="1,2"               # comma-separated, or empty = refresh all
```

> **Tip:** `arr_hook.sh` uses `curl` only — no Python needed. Works in any arr container out of the box.

### Manual submission

```bash
# Tab-complete the path — no quotes needed
/data/plugin/manual_bleeper.sh /data/media/movies/Film\ \(2024\)/Film\ \(2024\).mkv
```

`manual_bleeper.sh` extracts the movie title from the parent directory and delegates to `arr_hook.sh`, so Plex config is shared.

### Drop folder watcher

```bash
WATCH_FOLDER=/data/media/incoming python watcher.py
```

---

## Configuration

Key settings in `app/bleeper_backend.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `UPLOAD_FOLDER` | `$BLEEPER_UPLOAD` / `/app/uploads` | Staging area for temp files |
| `NORMALIZE_TARGET_CODEC` | `"ac3"` | Target codec. Set to `None` to skip normalization. |
| `NORMALIZE_TARGET_CHANNELS` | `6` | Target channel count (6 = 5.1) |
| `NORMALIZE_TARGET_BITRATE` | `"640k"` | Target audio bitrate |
| `FILTER_LIST_PATH` | `app/filter_list.txt` | Path to profanity word list |

---

## API Reference

### `POST /api/process_full`
Run the entire pipeline in one call (fire-and-forget).
```json
{
  "filename": "/data/media/movies/Film (2024)/Film (2024).mkv",
  "whisperx_settings": {
    "model": "large-v3",
    "language": "en",
    "batch_size": 16,
    "compute_type": "float16"
  },
  "plex_url": "http://plex:32400",
  "plex_token": "abc123",
  "plex_section_id": "1,2"
}
```
Returns HTTP 202:
```json
{
  "status": "queued",
  "job_id": "...",
  "position": 1,
  "message": "Pipeline started. Poll /api/job_status/<job_id> for progress."
}
```

### `GET /api/job_status/<job_id>`
Poll pipeline progress.
```json
{
  "job_id": "...",
  "status": "running",
  "stage": "transcribe",
  "error": "",
  "queue_position": null,
  "queue_depth": 2
}
```
`queue_position`: `null` = currently running or done; `N` = waiting in queue.

Status values: `queued` → `running` → `completed` | `failed`

### `POST /api/initialize_job`
Create a new job. Returns `{"job_id": "..."}`.

### `POST /api/select_remote_file`
Associate a file with a job.
```json
{"job_id": "...", "filename": "/data/media/movies/Film (2024)/Film (2024).mkv"}
```

### `POST /api/upload`
Upload a file (multipart/form-data). Fields: `file`, `job_id`.

### `POST /api/analyze_and_select_audio`
Probe and select best audio stream. `{"job_id": "..."}`

### `POST /api/normalize_audio`
Transcode to AC3 5.1. `{"job_id": "...", "force": false}`

### `POST /api/extract_audio`
Extract FC channel. `{"job_id": "..."}`

### `POST /api/transcribe`
Run whisperX. `{"job_id": "..."}`

### `POST /api/redact`
Mute + bleep profanity. `{"job_id": "..."}`

### `POST /api/combine_media`
Rebuild MKV. `{"job_id": "..."}`

### `POST /api/cleanup`
Move output to original path, remove temp files. `{"job_id": "..."}`

### `GET /api/list_files`
List video files in the upload staging folder.

---

## Output

The final MKV replaces the original file in-place and contains:

| Stream | Default | Description |
|---|---|---|
| Video | — | Original, untouched (stream copy) |
| Audio 1 | ✅ | **Family audio** – profanity muted/bleeped, AC3 5.1 |
| Audio 2 | ❌ | **Original audio** – untouched |
| Subtitles (existing) | ❌ | Original SRT/PGS/ASS streams copied as-is (`mov_text` stripped) |
| Subtitles (redacted) | ✅ | **Redacted subtitles** – profanity replaced with `***` |
