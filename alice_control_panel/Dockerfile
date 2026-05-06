ARG BUILD_FROM=python:3.12-slim
FROM ${BUILD_FROM}

LABEL \
  io.hass.version="0.1.2" \
  io.hass.type="addon" \
  io.hass.arch="amd64"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY static /app/static

ENV ALICE_STATIC_DIR=/app/static
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "app.main"]
