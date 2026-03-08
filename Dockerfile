FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SPACE_MODEL_DIR=/app/model \
    SPACE_ENABLED_MODELS=dinov2 \
    SPACE_DEFAULT_MODEL=dinov2

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    wget \
    procps \
    python3 \
    python3-pip \
    python3-venv \
    git \
    git-lfs \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user && mkdir -p /app && chown user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /app

COPY --chown=user requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

COPY --chown=user . /app

EXPOSE 7860
CMD ["python3", "-m", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
