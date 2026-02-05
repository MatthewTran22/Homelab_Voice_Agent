"""Unit tests for Discord bot."""

import struct
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

import pytest


class TestFilteredOutput:
    """Tests for FilteredOutput class."""

    def test_filters_opus_messages(self):
        from bot import FilteredOutput

        mock_original = Mock()
        filtered = FilteredOutput(mock_original)

        filtered.write("opus error occurred")
        mock_original.write.assert_not_called()

    def test_filters_decode_messages(self):
        from bot import FilteredOutput

        mock_original = Mock()
        filtered = FilteredOutput(mock_original)

        filtered.write("decode failed")
        mock_original.write.assert_not_called()

    def test_passes_normal_messages(self):
        from bot import FilteredOutput

        mock_original = Mock()
        filtered = FilteredOutput(mock_original)

        filtered.write("normal log message")
        mock_original.write.assert_called_once_with("normal log message")

    def test_flush_calls_original(self):
        from bot import FilteredOutput

        mock_original = Mock()
        filtered = FilteredOutput(mock_original)

        filtered.flush()
        mock_original.flush.assert_called_once()


class TestVADAudioSink:
    """Tests for VADAudioSink class."""

    @pytest.fixture
    def sink(self):
        with patch('bot.discord.opus.load_opus'):
            with patch('bot.webrtcvad.Vad'):
                from bot import VADAudioSink
                return VADAudioSink()

    def test_stereo_to_mono_conversion(self, sink):
        # Create stereo audio: left=1000, right=2000
        left = struct.pack("<h", 1000)
        right = struct.pack("<h", 2000)
        stereo = left + right

        mono = sink._stereo_to_mono(stereo)

        # Should average to 1500
        result = struct.unpack("<h", mono)[0]
        assert result == 1500

    def test_stereo_to_mono_multiple_samples(self, sink):
        # Two stereo samples
        sample1 = struct.pack("<h", 100) + struct.pack("<h", 200)  # avg=150
        sample2 = struct.pack("<h", 300) + struct.pack("<h", 400)  # avg=350
        stereo = sample1 + sample2

        mono = sink._stereo_to_mono(stereo)

        results = struct.unpack("<2h", mono)
        assert results == (150, 350)

    def test_stereo_to_mono_silence(self, sink):
        # Silence (zeros)
        stereo = b'\x00\x00\x00\x00'

        mono = sink._stereo_to_mono(stereo)

        result = struct.unpack("<h", mono)[0]
        assert result == 0

    def test_write_ignores_none_user(self, sink):
        sink.write(b"audio data", None)
        assert len(sink.user_buffers) == 0

    def test_write_ignores_none_data(self, sink):
        sink.write(None, 12345)
        assert len(sink.user_buffers) == 0

    def test_write_increments_frame_count(self, sink):
        user_id = 12345
        sink.write(b"x" * 100, user_id)
        assert sink.user_frame_count[user_id] == 1

    def test_user_buffers_isolated(self, sink):
        # Different users should have separate buffers
        sink.user_speaking[111] = True
        sink.user_speaking[222] = True

        sink.write(b"audio1", 111)
        sink.write(b"audio2", 222)

        assert sink.user_buffers[111] == [b"audio1"]
        assert sink.user_buffers[222] == [b"audio2"]

    def test_check_speech_returns_false_for_short_data(self, sink):
        result = sink._check_speech(b"short")
        assert result is False

    def test_check_speech_handles_exception(self, sink):
        sink.vad.is_speech = Mock(side_effect=Exception("VAD error"))

        # Should return False, not raise
        result = sink._check_speech(b"x" * 2000)
        assert result is False

    def test_cleanup_saves_remaining_audio(self, sink):
        sink.user_buffers[123] = [b"remaining audio"]
        sink._save_utterance = Mock()

        sink.cleanup()

        sink._save_utterance.assert_called_once_with(123)

    def test_cleanup_handles_empty_buffers(self, sink):
        sink.user_buffers[123] = []
        sink._save_utterance = Mock()

        sink.cleanup()

        # Should still be called but _save_utterance will early return
        sink._save_utterance.assert_called_once()


class TestSaveUtterance:
    """Tests for saving audio files."""

    @pytest.fixture
    def sink(self):
        with patch('bot.discord.opus.load_opus'):
            with patch('bot.webrtcvad.Vad'):
                from bot import VADAudioSink
                return VADAudioSink()

    def test_save_utterance_creates_wav_file(self, sink):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('bot.AUDIO_OUTPUT_DIR', tmpdir):
                # Add some audio data
                sink.user_buffers[123] = [b'\x00\x00' * 1000]

                sink._save_utterance(123)

                # Check a wav file was created
                files = os.listdir(tmpdir)
                assert len(files) == 1
                assert files[0].startswith("user_123_")
                assert files[0].endswith(".wav")

    def test_save_utterance_empty_buffer_noop(self, sink):
        sink.user_buffers[123] = []

        # Should not raise
        sink._save_utterance(123)


class TestConfigValues:
    """Tests for configuration constants."""

    def test_sample_rate(self):
        from bot import SAMPLE_RATE
        assert SAMPLE_RATE == 48000

    def test_channels(self):
        from bot import CHANNELS
        assert CHANNELS == 2

    def test_vad_aggressiveness_in_range(self):
        from bot import VAD_AGGRESSIVENESS
        assert 0 <= VAD_AGGRESSIVENESS <= 3

    def test_frame_calculations(self):
        from bot import (
            SAMPLE_RATE, FRAME_DURATION_MS, SAMPLES_PER_FRAME,
            BYTES_PER_SAMPLE, CHANNELS, FRAME_SIZE
        )

        expected_samples = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
        assert SAMPLES_PER_FRAME == expected_samples

        expected_size = expected_samples * CHANNELS * BYTES_PER_SAMPLE
        assert FRAME_SIZE == expected_size
