FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir openai-whisper grpcio grpcio-tools grpcio-reflection

# Pre-download Whisper small model during build
RUN python -c "import whisper; whisper.load_model('small')"

# Copy proto and generate Python code
COPY protos/asr.proto ./protos/
RUN python -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/asr.proto

COPY asr-agent/transcribe.py asr-agent/server.py ./

ENV PYTHONUNBUFFERED=1

EXPOSE 50051

CMD ["python", "server.py"]
