FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Configuración inicial
WORKDIR /app
COPY requirements.txt .

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Instalar numpy primero para evitar conflictos
RUN pip install --no-cache-dir numpy==1.23.5

# Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app
COPY . .

CMD ["python3", "auto_story_pipeline_chatgpt.py"]
