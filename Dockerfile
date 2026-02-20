# ─────────────────────────────────────────────────────────────────────────────
# Bleeper – Multi-architecture Dockerfile
#
# Supported platforms:
#   linux/amd64  – NVIDIA CUDA 12.1 GPU acceleration (full speed)
#   linux/arm64  – CPU-only (Raspberry Pi 4/5, Apple Silicon, ARM NAS)
#
# Build locally:
#   # Native arch only (fast):
#   docker build -t bleeper .
#
#   # Multi-arch (requires buildx + QEMU):
#   docker buildx build --platform linux/amd64,linux/arm64 -t bleeper --push .
#
# Run (GPU, amd64):
#   docker run --gpus all -p 5000:5000 \
#     -v /mnt/user/media:/media \
#     -v /mnt/user/appdata/bleeper/uploads:/app/uploads \
#     bleeper
#
# Run (CPU, arm64):
#   docker run -p 5000:5000 \
#     -e WHISPERX_COMPUTE_TYPE=int8 \
#     -v /mnt/user/media:/media \
#     -v /mnt/user/appdata/bleeper/uploads:/app/uploads \
#     bleeper
# ─────────────────────────────────────────────────────────────────────────────

ARG TARGETARCH

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1a: amd64 base — NVIDIA CUDA for GPU-accelerated whisperX
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base-amd64

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip

# PyTorch with CUDA 12.1 wheels (amd64 only)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    torch==2.3.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir whisperx>=3.1.1 \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b: arm64 base — Ubuntu + CPU-only PyTorch
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04 AS base-arm64

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip

# PyTorch CPU-only (arm64 — no CUDA wheels available)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    torch==2.3.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir whisperx>=3.1.1 \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Final image — pick the right base via TARGETARCH
# ─────────────────────────────────────────────────────────────────────────────
FROM base-${TARGETARCH} AS final

WORKDIR /app

COPY . .

RUN mkdir -p /app/uploads /media

ENV BLEEPER_UPLOAD=/app/uploads \
    WATCH_FOLDER=/media/incoming \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:/usr/local/sbin:${PATH}"

EXPOSE 5000

# Single gunicorn worker — job_status is in-process.
# Increase --timeout for long whisperX transcription jobs.
CMD ["python", "-m", "gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "600", "run:app"]
