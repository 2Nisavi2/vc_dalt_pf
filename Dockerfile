# Imagen de VisionVial para Railway (o cualquier plataforma de contenedores).
# Railway usa este Dockerfile automáticamente al encontrarlo en la raíz del repositorio.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: requerida por onnxruntime en imágenes slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Código, modelo registrado, métricas y capturas (ver .dockerignore)
COPY . .

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Railway define PORT en tiempo de ejecución; en local se usa 8501
EXPOSE 8501
CMD ["sh", "scripts/start.sh"]
