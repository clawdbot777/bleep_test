# ─────────────────────────────────────────────────────────────────────────────
# Bleeper – Automated Profanity Filter Pipeline
# Base: NVIDIA CUDA + Python 3.11 (GPU-accelerated whisperX transcription)
#
# Build:
#   docker build -t bleeper .
#
# Run (GPU):
#   docker run --gpus all -p 5000:5000 \
#     -v /mnt/user/media:/media \
#     -v /mnt/user/appdata/bleeper/uploads:/app/uploads \
#     bleeper
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip

# ── App dir ───────────────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
# Copy requirements first for better layer caching
COPY requirements.txt .

# PyTorch with CUDA 12.1 – install before whisperX so it picks up the GPU build
RUN pip install --no-cache-dir \
    torch==2.3.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# whisperX + remaining deps
RUN pip install --no-cache-dir whisperx>=3.1.1

RUN pip install --no-cache-dir -r requirements.txt

# ── App source ────────────────────────────────────────────────────────────────
COPY . .

# ── Runtime dirs ─────────────────────────────────────────────────────────────
# /app/uploads  – staging area for incoming media files
# /media        – mount point for your media library (Plex/Arr paths)
RUN mkdir -p /app/uploads /media

# ── Environment defaults ──────────────────────────────────────────────────────
ENV BLEEPER_UPLOAD=/app/uploads \
    WATCH_FOLDER=/media/incoming \
    PYTHONUNBUFFERED=1

# ── Expose ───────────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Entrypoint ───────────────────────────────────────────────────────────────
# Single gunicorn worker required – job_status lives in-process.
# Increase --timeout for long whisperX transcription jobs.
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "600", "run:app"]
