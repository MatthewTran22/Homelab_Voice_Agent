FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and Python packages in one layer
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir openai-whisper

# Pre-download Whisper small model during build (avoids runtime download)
RUN python -c "import whisper; whisper.load_model('small')"

COPY transcribe.py .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "transcribe.py"]
CMD ["--watch", "/app/audio"]
