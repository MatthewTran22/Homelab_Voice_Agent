FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (cached unless Dockerfile changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    ffmpeg \
    libffi-dev \
    libopus0 \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python deps first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code last (changes most often)
COPY bot.py .

RUN mkdir -p /app/audio

CMD ["python", "bot.py"]
