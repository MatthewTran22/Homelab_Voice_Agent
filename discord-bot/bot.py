"""Discord bot with Voice Activity Detection for audio capture."""

import os
import sys
import io
import wave
import struct
import asyncio
import logging
from datetime import datetime
from collections import defaultdict

# Suppress ALL discord logging before import
logging.getLogger("discord").setLevel(logging.CRITICAL)

import discord
from discord.ext import commands
import webrtcvad

# Monkey-patch to suppress opus decode errors on both stdout and stderr
class FilteredOutput:
    def __init__(self, original):
        self._original = original
    def write(self, msg):
        if "opus" not in msg.lower() and "decod" not in msg.lower() and "error occurred" not in msg.lower():
            self._original.write(msg)
    def flush(self):
        self._original.flush()

sys.stderr = FilteredOutput(sys.stderr)
sys.stdout = FilteredOutput(sys.stdout)

# Load opus library for voice
discord.opus.load_opus("libopus.so.0")

# Config
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
AUTO_JOIN_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")  # Voice channel to auto-join
AUDIO_OUTPUT_DIR = "/app/audio"
SAMPLE_RATE = 48000  # Discord default
CHANNELS = 2  # Discord sends stereo
FRAME_DURATION_MS = 20  # webrtcvad supports 10, 20, or 30ms
SILENCE_THRESHOLD_MS = 500  # How long silence before we consider speech ended
MIN_SPEECH_DURATION_MS = 200  # Ignore very short sounds
VAD_AGGRESSIVENESS = 3  # 0-3, higher = more aggressive filtering

# Calculate frame sizes
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
BYTES_PER_SAMPLE = 2  # 16-bit audio
FRAME_SIZE = SAMPLES_PER_FRAME * CHANNELS * BYTES_PER_SAMPLE
SILENCE_FRAMES = int(SILENCE_THRESHOLD_MS / FRAME_DURATION_MS)
MIN_SPEECH_FRAMES = int(MIN_SPEECH_DURATION_MS / FRAME_DURATION_MS)


class VADAudioSink(discord.sinks.Sink):
    """Custom sink that uses VAD to detect speech and save utterances."""

    def __init__(self):
        super().__init__()
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        # Per-user state
        self.user_buffers = defaultdict(list)
        self.user_speaking = defaultdict(bool)
        self.user_silence_frames = defaultdict(int)
        self.user_frame_count = defaultdict(int)
        # Only check VAD every N frames to reduce processing
        self.vad_check_interval = 5

    def write(self, data, user_id):
        """Called for each audio packet from a user."""
        if user_id is None or data is None:
            return

        self.user_frame_count[user_id] += 1

        # Only buffer if already speaking
        if self.user_speaking[user_id]:
            self.user_buffers[user_id].append(data)

        # Only check VAD every N frames
        if self.user_frame_count[user_id] % self.vad_check_interval != 0:
            return

        # Convert stereo to mono for VAD
        mono_data = self._stereo_to_mono(data)
        is_speech = self._check_speech(mono_data)

        if is_speech:
            if not self.user_speaking[user_id]:
                print(f"[user_{user_id}] Speaking...")
                self.user_speaking[user_id] = True
            # Buffer this frame too if we just started
            if data not in self.user_buffers[user_id]:
                self.user_buffers[user_id].append(data)
            self.user_silence_frames[user_id] = 0
        elif self.user_speaking[user_id]:
            self.user_silence_frames[user_id] += 1

            if self.user_silence_frames[user_id] >= (SILENCE_FRAMES // self.vad_check_interval):
                print(f"[user_{user_id}] Silent.")
                # Speech ended - save wav
                if len(self.user_buffers[user_id]) >= MIN_SPEECH_FRAMES:
                    self._save_utterance(user_id)

                # Reset for next utterance
                self.user_buffers[user_id] = []
                self.user_speaking[user_id] = False
                self.user_silence_frames[user_id] = 0

    def _stereo_to_mono(self, data):
        """Convert stereo audio to mono by averaging channels."""
        # Data is 16-bit stereo, so 4 bytes per sample pair
        samples = len(data) // 4
        mono = []
        for i in range(samples):
            left = struct.unpack_from("<h", data, i * 4)[0]
            right = struct.unpack_from("<h", data, i * 4 + 2)[0]
            mono_sample = (left + right) // 2
            mono.append(struct.pack("<h", mono_sample))
        return b"".join(mono)

    def _check_speech(self, mono_data):
        """Check if audio frame contains speech using webrtcvad."""
        # webrtcvad needs exactly 10, 20, or 30ms of audio
        # At 48kHz, 20ms = 960 samples = 1920 bytes (16-bit)
        frame_bytes = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2

        if len(mono_data) < frame_bytes:
            return False

        frame = mono_data[:frame_bytes]
        try:
            return self.vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            # Silently ignore decode errors
            return False

    def _save_utterance(self, user_id):
        """Save buffered audio as a wav file."""
        if not self.user_buffers[user_id]:
            return

        # Combine all buffered audio
        audio_data = b"".join(self.user_buffers[user_id])

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{AUDIO_OUTPUT_DIR}/user_{user_id}_{timestamp}.wav"

        # Write wav file
        os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)

        duration = len(audio_data) / (SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE)
        print(f"Saved utterance: {filename} ({duration:.2f}s)")

    def cleanup(self):
        """Save any remaining audio when recording stops."""
        for user_id in list(self.user_buffers.keys()):
            if self.user_buffers[user_id]:
                self._save_utterance(user_id)


# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    """Called when bot is ready."""
    print(f"Bot is ready! Logged in as {bot.user}")

    # Auto-join voice channel if configured
    if AUTO_JOIN_CHANNEL_ID:
        await auto_join_channel()


async def auto_join_channel():
    """Automatically join the configured voice channel."""
    try:
        channel_id = int(AUTO_JOIN_CHANNEL_ID)
        channel = bot.get_channel(channel_id)

        if channel is None:
            print(f"Error: Could not find channel with ID {channel_id}")
            return

        if not isinstance(channel, discord.VoiceChannel):
            print(f"Error: Channel {channel_id} is not a voice channel")
            return

        voice_client = await channel.connect()
        print(f"Auto-joined voice channel: {channel.name}")

        # Start recording with VAD
        voice_client.start_recording(
            VADAudioSink(),
            auto_finished_callback,
            channel,
        )
        print("Listening with VAD enabled...")

    except Exception as e:
        print(f"Error auto-joining channel: {e}")


async def auto_finished_callback(sink, channel):
    """Called when auto recording is stopped."""
    sink.cleanup()
    print("Stopped listening.")


@bot.command()
async def join(ctx):
    """Join the user's voice channel."""
    if ctx.author.voice is None:
        await ctx.send("You need to be in a voice channel!")
        return

    channel = ctx.author.voice.channel

    if ctx.voice_client is not None:
        await ctx.voice_client.move_to(channel)
    else:
        await channel.connect()

    await ctx.send(f"Joined {channel.name}!")

    # Start recording with VAD sink
    ctx.voice_client.start_recording(
        VADAudioSink(),
        finished_callback,
        ctx.channel,
    )
    await ctx.send("Listening with VAD enabled...")


async def finished_callback(sink, channel):
    """Called when recording is stopped."""
    sink.cleanup()
    await channel.send("Stopped listening.")


@bot.command()
async def leave(ctx):
    """Leave the voice channel."""
    if ctx.voice_client is None:
        await ctx.send("I'm not in a voice channel!")
        return

    ctx.voice_client.stop_recording()
    await ctx.voice_client.disconnect()
    await ctx.send("Left the voice channel!")


@bot.command()
async def status(ctx):
    """Check bot status."""
    if ctx.voice_client and ctx.voice_client.is_connected():
        await ctx.send(f"Connected to: {ctx.voice_client.channel.name}")
    else:
        await ctx.send("Not connected to any voice channel.")


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("Error: DISCORD_BOT_TOKEN environment variable not set!")
        exit(1)

    print("Starting Discord bot...")
    bot.run(DISCORD_TOKEN)
