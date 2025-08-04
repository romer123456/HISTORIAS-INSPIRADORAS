# Inspirador (YouTube Auto Pipeline)

## Requisitos
- OpenAI API key (export OPENAI_API_KEY)
- Docker
- (Opcional GPU) CUDA image distinta

## Construir imagen
docker build -t inspirador:latest .

## Ejecutar (CPU)
docker run --rm -e OPENAI_API_KEY="TU_API_KEY" \
  -v $(pwd)/outputs:/app/outputs \
  inspirador:latest \
  --youtube_url "https://www.youtube.com/watch?v=XXXXXXXX" \
  --instruction "Escribe una historia inspiradora de 15 minutos, tono cinematográfico, en español."

# Salida en outputs/<ID>/
