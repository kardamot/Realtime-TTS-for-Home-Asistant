FROM python:3.12-slim

LABEL \
      io.hass.version="0.1.41" \
  io.hass.type="addon" \
  io.hass.arch="amd64"

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY static /app/static

ENV ALICE_STATIC_DIR=/app/static
ENV ALICE_MODELS_DIR=/data/models
ENV HF_HOME=/data/models/huggingface
ENV HF_HUB_CACHE=/data/models/huggingface/hub
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "app.main"]
