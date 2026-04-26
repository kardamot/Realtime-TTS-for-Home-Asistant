FROM python:3.12-alpine

LABEL \
  io.hass.version="0.4.0" \
  io.hass.type="addon" \
  io.hass.arch="amd64"

RUN pip install --no-cache-dir aiohttp google-auth requests

COPY tts_relay_server.py /app/tts_relay_server.py

CMD ["python3", "/app/tts_relay_server.py"]
