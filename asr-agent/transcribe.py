#!/usr/bin/env python3
"""
ASR Agent - Transcribes WAV files using OpenAI Whisper tiny model.

Usage:
    python transcribe.py <path_to_wav_file>
    python transcribe.py --watch <directory>  # Watch directory for new files
"""

import argparse
import os
import sys
import time
from pathlib import Path

import whisper


def transcribe_file(model, file_path: str) -> str:
    """Transcribe a single WAV file and return the text."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return ""

    print(f"Transcribing: {file_path}")
    result = model.transcribe(file_path)
    return result["text"]


def watch_directory(model, directory: str, processed_files: set):
    """Watch a directory for new WAV files and transcribe them."""
    path = Path(directory)

    for wav_file in path.glob("*.wav"):
        file_path = str(wav_file)
        if file_path not in processed_files:
            text = transcribe_file(model, file_path)
            if text:
                print(f"\n{'='*60}")
                print(f"File: {wav_file.name}")
                print(f"Transcription: {text.strip()}")
                print(f"{'='*60}\n")
            processed_files.add(file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe WAV files using Whisper tiny model"
    )
    parser.add_argument(
        "input",
        help="Path to WAV file or directory (with --watch)"
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Watch directory for new WAV files continuously"
    )
    parser.add_argument(
        "--model", "-m",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)"
    )

    args = parser.parse_args()

    print(f"Loading Whisper {args.model} model...")
    model = whisper.load_model(args.model)
    print("Model loaded successfully!")

    if args.watch:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory", file=sys.stderr)
            sys.exit(1)

        print(f"Watching directory: {args.input}")
        print("Press Ctrl+C to stop\n")

        processed_files = set()

        try:
            while True:
                watch_directory(model, args.input, processed_files)
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping watcher...")
    else:
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} is not a file", file=sys.stderr)
            sys.exit(1)

        text = transcribe_file(model, args.input)
        if text:
            print(f"\nTranscription: {text.strip()}\n")


if __name__ == "__main__":
    main()
