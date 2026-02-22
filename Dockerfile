# ─────────────────────────────────────────────────────────────────────────────
# Bleeper – Automated Profanity Filter Pipeline
# Base: NVIDIA CUDA 12.1 + Python 3.11 (GPU-accelerated faster-whisper)
#
# Build:
#   docker build -t bleeper .
#
# Run:
#   docker run --gpus all -p 5000:5000 \
#     -v /mnt/user/media:/media \
#     -v /mnt/user/appdata/bleeper/uploads:/app/uploads \
#     bleeper
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg git curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip

ENV PATH="/usr/local/bin:/usr/local/sbin:${PATH}"

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.4.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir faster-whisper \
 && pip install --no-cache-dir -r requirements.txt

# ── App ───────────────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .
RUN mkdir -p /app/uploads /app/models /media

ENV BLEEPER_UPLOAD=/app/uploads \
    MODEL_DIR=/app/models \
    WATCH_FOLDER=/media/incoming \
    PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "-m", "gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "600", "run:app"]
