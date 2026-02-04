# Agent Architecture

## Ubuntu Laptop

| Agent | Description |
|-------|-------------|
| **Discord Bot** | Joins voice channels, captures audio with VAD, saves WAV files |
| **LLM Agent** | Processes transcriptions and generates responses |

## Jetson Orin Nano

| Agent | Description |
|-------|-------------|
| **ASR Agent** | Transcribes WAV files using Whisper small model |
| **TTS Agent** | Converts text responses to speech audio |

## Data Flow

```
Discord Voice → Discord Bot (Ubuntu) → WAV files
                                           ↓
                                    ASR Agent (Jetson) → Transcription
                                                              ↓
                                                      LLM Agent (Ubuntu) → Response
                                                              ↓
                                                      TTS Agent (Jetson) → Audio
                                                              ↓
                                                      Discord Bot (Ubuntu) → Voice Output
```
