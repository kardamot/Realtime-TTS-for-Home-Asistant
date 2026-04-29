FROM python:3.12-slim

LABEL \
  io.hass.version="0.9.17" \
  io.hass.type="addon" \
  io.hass.arch="amd64"

RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY voice_server.py /app/voice_server.py

CMD ["python3", "/app/voice_server.py"]
