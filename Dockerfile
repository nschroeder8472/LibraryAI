FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY scripts/ scripts/
COPY main.py .
RUN mkdir -p /app/data/raw /app/data/processed /app/data/vector_store

ENV HF_HOME=/models
ENV AUTO_DETECT_DEVICE=true
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
