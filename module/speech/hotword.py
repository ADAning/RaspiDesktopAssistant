"""
For the Raspberry Pi desktop assistant's hot word detection functionality. Other input monitoring components (Vision, LLM, etc.) are activated only when the wake word is triggered.
"""

import logging
import threading
import time
from typing import Callable, Optional
import os
import struct
import pvporcupine
import pyaudio
from config_loader import config

logger = logging.getLogger(__name__)


class HotwordDetector:
    """Hotword detector, used to listen for wake-up words"""

    def __init__(
        self,
        callback: Optional[Callable[[], None]] = None,
    ):
        """
        Initializing hotword detector

        Args:
            callback: Callback function when hotword is detected
        """
        self.callback = callback

        hot_word_config = config("speech.hot_word") or {}
        self.access_key = hot_word_config.get("access_key")
        self.keywords = hot_word_config.get("keywords", ["hey siri"])
        self.sensitivities = hot_word_config.get(
            "sensitivities", [0.5] * len(self.keywords)
        )

        logger.info(f"Initializing hotword detector with keywords: {self.keywords}")
        logger.info(f"Sensitivities: {self.sensitivities}")

        if not self.access_key:
            self.access_key = os.getenv("PICOVOICE_ACCESS_KEY")
            if not self.access_key:
                raise ValueError(
                    "Porcupine access key is required. Please set it in config.yaml or PICOVOICE_ACCESS_KEY environment variable."
                )

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._porcupine = None
        self._audio = None
        self._audio_stream = None
        logger.info("Hotword detection module initialized")

    def _create_porcupine(self):
        """Create Porcupine instance with configured keywords"""
        try:
            logger.info("Creating Porcupine instance...")
            self._porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=self.keywords,
                sensitivities=self.sensitivities,
            )
            logger.info("Porcupine instance created successfully")

            self._audio = pyaudio.PyAudio()
            self._audio_stream = self._audio.open(
                rate=self._porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self._porcupine.frame_length,
            )
            logger.info("Audio stream initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {str(e)}")
            raise

    def _listen_for_hotword(self):
        """Internal method that runs in a separate thread to listen for the hotword."""
        try:
            self._create_porcupine()
            logger.info(
                f"Hotword listener thread started, listening for keywords: {self.keywords}"
            )

            while not self._stop_event.is_set():
                pcm = self._audio_stream.read(
                    self._porcupine.frame_length, exception_on_overflow=False
                )
                pcm = struct.unpack_from("h" * self._porcupine.frame_length, pcm)

                keyword_index = self._porcupine.process(pcm)
                if keyword_index >= 0:
                    detected_keyword = self.keywords[keyword_index]
                    logger.info(f"Detected wake word: '{detected_keyword}'!")
                    if self.callback:
                        try:
                            self.callback()
                        except Exception as e:
                            logger.error(
                                f"Error in hotword callback: {e}", exc_info=True
                            )

        except Exception as e:
            logger.error(f"Error in hotword detection: {str(e)}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources"""
        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()

        if self._audio:
            self._audio.terminate()

        if self._porcupine:
            self._porcupine.delete()

    def start(self):
        """Start listening for hotwords in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Hotword detector is already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_for_hotword)
        self._thread.daemon = True
        self._thread.start()
        logger.info("Hotword detector started")

    def stop(self):
        """Stop listening for hotwords."""
        if not self._thread or not self._thread.is_alive():
            logger.warning("Hotword detector is not running")
            return

        logger.info("Stopping hotword detector...")
        self._stop_event.set()
        self._thread.join(timeout=2)  # Wait for the thread to finish
        self._thread = None
        logger.info("Hotword detector stopped.")

    def is_active(self) -> bool:
        """
        Check if the hotword detector is active

        Returns:
            Whether the hotword detector is active
        """
        return self._thread is not None and self._thread.is_alive()
