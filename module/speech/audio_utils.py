"""
Shared audio utilities for speech recognition and synthesis.
"""

import os
import wave
import logging
import tempfile
import pyaudio
from typing import Optional, Tuple
from config_loader import config

logger = logging.getLogger(__name__)


class AudioManager:
    """Audio management class for handling recording and playback"""

    def __init__(
        self,
        channels: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ):
        """
        Initialize audio manager

        Args:
            channels: Number of audio channels, if None use config value
            sampling_rate: Sample rate in Hz, if None use config value
            chunk_size: Size of audio chunks for processing, if None use config value
        """
        # Load audio configuration
        audio_config = config("speech.audio") or {}

        # Standard format for most audio processing
        self.format = pyaudio.paInt16
        self.channels = channels or audio_config.get("channels", 1)
        self.sampling_rate = sampling_rate or audio_config.get("sampling_rate", 16000)
        self.chunk = chunk_size or audio_config.get("chunk_size", 1024)
        self.audio = pyaudio.PyAudio()

    def __del__(self):
        """Cleanup PyAudio when object is destroyed"""
        try:
            self.audio.terminate()
        except:
            pass

    def record_audio(self, duration: int) -> Tuple[bytes, str]:
        """
        Record audio from microphone

        Args:
            duration: Recording duration in seconds

        Returns:
            Tuple of (audio data in bytes, WAV file path)
        """
        try:
            # Open audio stream
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sampling_rate,
                input=True,
                frames_per_buffer=self.chunk,
            )

            logger.info(f"Recording audio for {duration} seconds...")
            frames = []

            # Record audio
            for _ in range(0, int(self.sampling_rate / self.chunk * duration)):
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)

            logger.info("Finished recording")

            # Stop and close the stream
            stream.stop_stream()
            stream.close()

            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                with wave.open(temp_wav.name, "wb") as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.format))
                    wf.setframerate(self.sampling_rate)
                    wf.writeframes(b"".join(frames))

                # Read the WAV file
                with open(temp_wav.name, "rb") as f:
                    audio_data = f.read()

            return audio_data, temp_wav.name

        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            raise

    def play_audio(self, audio_file: str) -> bool:
        """
        Play audio file

        Args:
            audio_file: Audio file path

        Returns:
            Whether the audio file is successfully played
        """
        try:
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return False

            # Open WAV file
            with wave.open(audio_file, "rb") as wf:
                # Create playback stream
                stream = self.audio.open(
                    format=self.audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                )

                try:
                    # Read and play data
                    data = wf.readframes(self.chunk)
                    while data:
                        stream.write(data)
                        data = wf.readframes(self.chunk)

                finally:
                    # Close stream
                    stream.stop_stream()
                    stream.close()

            return True

        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False

    @staticmethod
    def cleanup_temp_file(file_path: str) -> None:
        """
        Clean up temporary audio file

        Args:
            file_path: File path to clean up
        """
        try:
            if file_path.startswith(tempfile.gettempdir()):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary audio file: {e}")
