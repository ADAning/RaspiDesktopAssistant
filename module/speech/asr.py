"""
For the Raspberry Pi desktop assistant's speech recognition functionality, converting microphone audio input into text for large language model processing.
"""

import os
import time
import queue
import threading
import logging
from typing import Optional, Callable, Generator
from aip.speech import AipSpeech
from config_loader import config
from .audio_utils import AudioManager

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """Speech recognition class, converting audio to text using Baidu ASR API"""

    def __init__(self):
        """Initializing speech recognizer with Baidu ASR credentials"""
        # Initialize Baidu ASR client
        self.app_id = config("speech.asr.baidu.app_id")
        self.api_key = config("speech.asr.baidu.api_key")
        self.secret_key = config("speech.asr.baidu.secret_key")

        if not all([self.app_id, self.api_key, self.secret_key]):
            logger.error("Baidu ASR credentials not properly configured")
            raise ValueError("Missing Baidu ASR credentials")

        self.client = AipSpeech(self.app_id, self.api_key, self.secret_key)

        # Load configuration
        self.language = config("speech.asr.baidu.language") or "zh-CN"
        self.dev_pid = config("speech.asr.baidu.dev_pid") or 1537
        self.record_seconds = config("speech.asr.baidu.record_timeout") or 5

        # Streaming configuration
        self.stream_enabled = config("speech.asr.stream.enable") or False
        self.chunk_size = config("speech.asr.stream.chunk_size") or 2048
        self.max_silence = config("speech.asr.stream.max_silence") or 2.0
        self.min_speaking = config("speech.asr.stream.min_speaking") or 0.5
        self.energy_threshold = config("speech.asr.stream.energy_threshold") or 4000

        # Initialize audio manager with default configuration
        self.audio_manager = AudioManager()

        # Streaming support
        self.is_streaming = False
        self.stream_queue = queue.Queue()
        self.stream_thread = None

        logger.info("Speech recognition module initialized with Baidu ASR")

    def _record_stream(self) -> Generator[bytes, None, None]:
        """
        Record audio stream from microphone with voice activity detection

        Yields:
            Audio chunks as bytes
        """
        audio = self.audio_manager.audio
        stream = audio.open(
            format=self.audio_manager.format,
            channels=self.audio_manager.channels,
            rate=self.audio_manager.sampling_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        try:
            silence_start = None
            is_speaking = False
            speaking_start = None
            frames = []

            logger.info("Started streaming audio recording")
            while self.is_streaming:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)

                # Calculate audio energy
                energy = (
                    sum(
                        abs(int.from_bytes(data[i : i + 2], "little", signed=True))
                        for i in range(0, len(data), 2)
                    )
                    / self.chunk_size
                )

                # Voice activity detection
                if energy > self.energy_threshold:
                    if not is_speaking:
                        speaking_start = time.time()
                        is_speaking = True
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.time()

                # Check if we've been speaking long enough
                if (
                    is_speaking
                    and speaking_start
                    and time.time() - speaking_start >= self.min_speaking
                ):
                    yield b"".join(
                        frames[
                            -int(
                                self.min_speaking
                                * self.audio_manager.sampling_rate
                                / self.chunk_size
                            ) :
                        ]
                    )
                    frames = []
                    is_speaking = False
                    speaking_start = None

                # Check if silence duration exceeds threshold
                if silence_start and time.time() - silence_start >= self.max_silence:
                    break

        finally:
            stream.stop_stream()
            stream.close()
            logger.info("Stopped streaming audio recording")

    def _process_stream(self) -> None:
        """Process audio stream and put recognition results in queue"""
        try:
            for audio_chunk in self._record_stream():
                # Send to Baidu ASR
                result = self.client.asr(
                    audio_chunk,
                    "wav",
                    self.audio_manager.sampling_rate,
                    {
                        "dev_pid": self.dev_pid,
                    },
                )

                if result["err_no"] == 0:
                    recognized_text = result["result"][0]
                    logger.info(f"Stream recognized: {recognized_text}")
                    self.stream_queue.put(recognized_text)
                else:
                    logger.debug(f"Stream recognition failed: {result}")
        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
        finally:
            self.is_streaming = False

    def start_streaming(self) -> None:
        """Start streaming recognition"""
        if not self.stream_enabled:
            logger.warning("Stream recognition is disabled in configuration")
            return

        if self.is_streaming:
            logger.warning("Streaming is already active")
            return

        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._process_stream)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        logger.info("Started streaming recognition")

    def stop_streaming(self) -> None:
        """Stop streaming recognition"""
        if not self.is_streaming:
            return

        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
            self.stream_thread = None
        logger.info("Stopped streaming recognition")

    def get_streaming_result(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Get next recognition result from streaming

        Args:
            timeout: How long to wait for next result (seconds), None for no timeout

        Returns:
            Recognition result text, or None if no result available
        """
        try:
            return self.stream_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def recognize_from_microphone(self, timeout: Optional[int] = None) -> str:
        """
        Record audio from microphone and recognize it as text

        Args:
            timeout: Recording timeout (seconds), if None use default

        Returns:
            Recognition result text
        """
        try:
            # Record audio
            audio_data, temp_file = self.audio_manager.record_audio(
                timeout or self.record_seconds
            )

            try:
                # Send to Baidu ASR
                result = self.client.asr(
                    audio_data,
                    "wav",
                    self.audio_manager.sampling_rate,
                    {
                        "dev_pid": self.dev_pid,
                    },
                )

                if result["err_no"] == 0:
                    recognized_text = result["result"][0]
                    logger.info(f"Successfully recognized: {recognized_text}")
                    return recognized_text
                else:
                    logger.error(
                        f"Recognition failed with error {result['err_no']}: {result['err_msg']}"
                    )
                    return ""

            finally:
                # Clean up temporary file
                self.audio_manager.cleanup_temp_file(temp_file)

        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return ""

    def recognize_from_file(self, audio_file: str) -> str:
        """
        Recognize text from audio file

        Args:
            audio_file: Audio file path

        Returns:
            Recognition result text
        """
        try:
            # Read audio file
            with open(audio_file, "rb") as f:
                audio_data = f.read()

            # Send to Baidu ASR
            result = self.client.asr(
                audio_data,
                "wav",
                self.audio_manager.sampling_rate,
                {
                    "dev_pid": self.dev_pid,
                },
            )

            if result["err_no"] == 0:
                recognized_text = result["result"][0]
                logger.info(f"Successfully recognized from file: {recognized_text}")
                return recognized_text
            else:
                logger.error(
                    f"File recognition failed with error {result['err_no']}: {result['err_msg']}"
                )
                return ""
        except Exception as e:
            logger.error(f"Error in file recognition: {e}")
            return ""

    def listen_and_recognize(
        self,
        callback: Optional[Callable[[str], None]] = None,
        energy_threshold: int = 4000,
        pause_threshold: float = 0.8,
    ) -> str:
        """
        Listen to microphone and recognize when speech is detected

        Args:
            callback: Recognition result callback function
            energy_threshold: Energy threshold, used to detect speech start
            pause_threshold: Pause threshold, used to detect speech end

        Returns:
            Recognition result text
        """
        # For now, we'll use a simple implementation without VAD
        result = self.recognize_from_microphone()

        if callback and result:
            callback(result)

        return result
