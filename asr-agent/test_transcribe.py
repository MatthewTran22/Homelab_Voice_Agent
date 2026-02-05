"""Integration tests for ASR gRPC service."""

import re

import grpc
import pytest

import asr_pb2
import asr_pb2_grpc

ASR_HOST = "asr-agent:50051"

TEST_AUDIO_UNDEFINED = "/app/audio-tests/nonexistent_file.wav"
TEST_AUDIO_SPEECH = "/app/audio-tests/voice-active.mp3"

EXPECTED_SPEECH = "they say behind every great man is a great woman but lets be honest im usually three steps ahead"


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@pytest.fixture(scope="module")
def stub():
    """Create a gRPC stub connected to the ASR service."""
    channel = grpc.insecure_channel(ASR_HOST)
    grpc.channel_ready_future(channel).result(timeout=30)
    return asr_pb2_grpc.ASRServiceStub(channel)


class TestASRService:
    """Integration tests for the ASR gRPC service."""

    def test_undefined_file_path(self, stub):
        """Test that undefined file path returns error."""
        response = stub.Transcribe(
            asr_pb2.TranscribeRequest(file_path=TEST_AUDIO_UNDEFINED)
        )

        assert response.success is False
        assert "not found" in response.error.lower()

    def test_speech_audio_file(self, stub):
        """Test transcription of audio with speech matches expected text."""
        response = stub.Transcribe(
            asr_pb2.TranscribeRequest(file_path=TEST_AUDIO_SPEECH)
        )

        assert response.success is True
        assert response.error == ""
        normalized = normalize_text(response.text)
        assert normalized == EXPECTED_SPEECH
