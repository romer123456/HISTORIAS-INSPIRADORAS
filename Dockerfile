FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Copiamos requirements.txt primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Instalar numpy primero para evitar conflictos con TTS y librosa
RUN pip install --no-cache-dir numpy==1.22.0

# Instalar el resto de dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Comando por defecto para ejecutar tu script
CMD ["python3", "auto_story_pipeline_chatgpt.py"]
