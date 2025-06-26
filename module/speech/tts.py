"""
For the Raspberry Pi desktop assistant's text-to-speech functionality, converting LLM responses into spoken audio output.
"""

import os
import logging
import tempfile
from typing import Optional
from aip.speech import AipSpeech
from config_loader import config
from .audio_utils import AudioManager

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Text-to-speech class, converting text to speech and playing it using Baidu TTS API"""

    def __init__(self):
        """Initializing TTS engine with Baidu TTS credentials"""
        # Initialize Baidu TTS client
        self.app_id = config("speech.tts.baidu.app_id")
        self.api_key = config("speech.tts.baidu.api_key")
        self.secret_key = config("speech.tts.baidu.secret_key")

        if not all([self.app_id, self.api_key, self.secret_key]):
            logger.error("Baidu TTS credentials not properly configured")
            raise ValueError("Missing Baidu TTS credentials")

        self.client = AipSpeech(self.app_id, self.api_key, self.secret_key)

        # Load configuration
        self.language = config("speech.tts.baidu.language") or "zh"
        self.format = config("speech.tts.baidu.format") or 6  # WAV format

        # TTS parameters
        self.voice = config("speech.tts.baidu.voice") or 0
        self.speed = config("speech.tts.baidu.speed") or 5
        self.pitch = config("speech.tts.baidu.pitch") or 5
        self.volume = config("speech.tts.baidu.volume") or 5

        # Initialize audio manager with default configuration
        self.audio_manager = AudioManager()

        # Stream support
        self.buffer = ""
        self.stream_enabled = config("speech.tts.stream.enable") or False
        self.chunk_size = config("speech.tts.stream.chunk_size") or 2048
        self.sentence_end_marks = config("speech.tts.stream.sentence_end_marks") or [
            "。",
            "！",
            "？",
            ".",
            "!",
            "?",
            "\n",
        ]

        logger.info("Text-to-speech module initialized with Baidu TTS")

    def text_to_speech(
        self, text: str, language: Optional[str] = None
    ) -> Optional[str]:
        """
        Convert text to speech and save it as a temporary file

        Args:
            text: Text to convert
            language: Language code, if None then use default language

        Returns:
            Generated audio file path, if failed then return None
        """
        try:
            # If text is empty, return
            if not text.strip():
                logger.warning("Empty text provided for TTS")
                return None

            # Use temporary file to save audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                # Call Baidu TTS API
                result = self.client.synthesis(
                    text,
                    language or self.language,
                    self.format,
                    {
                        "per": self.voice,
                        "spd": self.speed,
                        "pit": self.pitch,
                        "vol": self.volume,
                        "aue": self.format,
                    },
                )

                # Check if the return result is audio data
                if not isinstance(result, dict):
                    # Write audio file
                    if isinstance(result, str):
                        result = result.encode("utf-8")
                    temp_audio.write(result)
                    logger.info(f"Generated audio file: {temp_audio.name}")
                    return temp_audio.name
                else:
                    logger.error(f"TTS synthesis failed: {result}")
                    return None

        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}")
            return None

    def speak(self, text: str, language: Optional[str] = None) -> bool:
        """
        Convert text to speech and play it

        Args:
            text: Text to convert
            language: Language code, if None then use default language

        Returns:
            Whether the text is successfully spoken
        """
        try:
            # Convert text to speech file
            audio_file = self.text_to_speech(text, language)
            if not audio_file:
                return False

            try:
                # Play speech file
                return self.audio_manager.play_audio(audio_file)
            finally:
                # Clean up temporary file
                self.audio_manager.cleanup_temp_file(audio_file)

        except Exception as e:
            logger.error(f"Error in speak function: {e}")
            return False

    def process_stream(self, text_chunk: str) -> bool:
        """
        Process streaming text chunk, when a complete sentence is detected, do text-to-speech

        Args:
            text_chunk: Text chunk to process

        Returns:
            Whether any text is processed and spoken
        """
        if not self.stream_enabled:
            logger.warning("Stream processing is disabled in configuration")
            return False

        if not text_chunk:
            return False

        self.buffer += text_chunk
        processed = False

        # Check if there is a complete sentence
        for mark in self.sentence_end_marks:
            while mark in self.buffer:
                # Split the first complete sentence
                parts = self.buffer.split(mark, 1)
                if len(parts) > 1:
                    sentence = parts[0] + mark
                    self.buffer = parts[1]

                    # Play this sentence
                    if sentence.strip():
                        self.speak(sentence)
                        processed = True
                else:
                    break

        return processed

    def flush_stream(self) -> bool:
        """
        Force processing all remaining text in the stream buffer

        Returns:
            Whether the text is successfully spoken
        """
        if not self.stream_enabled:
            logger.warning("Stream processing is disabled in configuration")
            return False

        if not self.buffer:
            return False

        # Process the remaining text in the buffer
        success = False
        if self.buffer.strip():
            success = self.speak(self.buffer)
        self.buffer = ""
        return success
