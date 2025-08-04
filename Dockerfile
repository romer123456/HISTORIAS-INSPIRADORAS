FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# Paquetes del sistema necesarios (ffmpeg, audio/TTS)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip git ffmpeg \
    libsndfile1 espeak-ng wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Torch con CUDA 12.1 (instalar antes que el resto)
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Dependencias del proyecto (NO pongas torch aquí)
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Entrada por defecto
# Copia código y carpetas
COPY auto_story_pipeline_chatgpt.py /app/auto_story_pipeline_chatgpt.py
COPY data /app/data
RUN mkdir -p /app/outputs

ENTRYPOINT ["python3", "/app/auto_story_pipeline_chatgpt.py"]
