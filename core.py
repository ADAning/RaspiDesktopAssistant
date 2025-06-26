import time
from logging import getLogger
import logging
from config_loader import config
from module.llm.api import LLM
from module.vision.camera import Camera
from module.speech.asr import SpeechRecognizer
from module.speech.tts import TextToSpeech
from module.speech.hotword import HotwordDetector
from typing import Dict, Any, Optional, Union


class Assistant:
    def __init__(self):
        # Setup logging
        self._setup_logging()
        self.logger = getLogger(__name__)
        self.logger.info("Initializing assistant...")

        # Initialize modules
        self.modules: Dict[str, Any] = {}
        self._init_modules()

    def _setup_logging(self):
        """Setup logging system using utility function"""
        log_level = config("system.log_level") or "INFO"
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        level = log_level_map.get(log_level, logging.INFO)

        # Create log directory
        log_dir = config("system.storage_path") or "./logs"
        os.makedirs(log_dir, exist_ok=True)

        # Configure log format
        log_file = os.path.join(log_dir, "assistant.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Set file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Set console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Get root logger and configure
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear all existing handlers
        root_logger.handlers.clear()

        # Add new handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def _init_modules(self):
        """Initialize all modules"""
        # Initialize LLM module
        try:
            self.modules["llm"] = self._init_llm()
        except Exception as e:
            self.logger.error(f"LLM module initialization failed: {e}")

        # Initialize vision module
        if config("vision.enable"):
            try:
                self.modules["camera"] = self._init_camera()
            except Exception as e:
                self.logger.error(f"Vision module initialization failed: {e}")

        # Initialize speech recognition module
        if config("speech.asr.enable"):
            try:
                self.modules["asr"] = self._init_speech_recognizer()
            except Exception as e:
                self.logger.error(
                    f"Speech recognition module initialization failed: {e}"
                )

        # Initialize speech synthesis module
        if config("speech.tts.enable"):
            try:
                self.modules["tts"] = self._init_text_to_speech()
            except Exception as e:
                self.logger.error(f"Speech synthesis module initialization failed: {e}")

        # Initialize hotword detection module
        try:
            self.modules["hotword"] = self._init_hotword_detector()
        except Exception as e:
            self.logger.error(f"Hotword detection module initialization failed: {e}")

        # Initialize action module if enabled
        if config("action.enable"):
            try:
                # TODO: Initialize action modules here
                pass
            except Exception as e:
                self.logger.error(f"Action module initialization failed: {e}")

    def _init_llm(self) -> LLM:
        """Initialize large language model"""
        # Text model configuration (required)
        text_config = config("llm.text") or {}
        text_api_key = text_config.get("api_key")
        text_base_url = text_config.get("base_url")

        # Use a single system prompt for both text and vision
        system_prompt = text_config.get("init_prompt") or "You are a helpful assistant."

        # Vision model configuration (optional)
        vision_config = config("llm.vision") or {}
        vision_enabled = vision_config.get("enable", False)
        vision_api_key = None
        vision_base_url = None

        if vision_enabled:
            vision_api_key = vision_config.get("api_key")
            vision_base_url = vision_config.get("base_url")
            self.logger.info("Vision model is enabled")
        else:
            self.logger.info("Vision model is disabled")

        return LLM(
            text_api_key=text_api_key,
            text_base_url=text_base_url,
            vision_api_key=vision_api_key,
            vision_base_url=vision_base_url,
            system_prompt=system_prompt,
        )

    def _init_camera(self) -> Camera:
        """Initialize camera module"""
        return Camera()

    def _init_speech_recognizer(self) -> SpeechRecognizer:
        """Initialize speech recognition module"""
        return SpeechRecognizer()

    def _init_text_to_speech(self) -> TextToSpeech:
        """Initialize text-to-speech module"""
        return TextToSpeech()

    def _init_hotword_detector(self) -> HotwordDetector:
        """Initialize hotword detection module"""
        return HotwordDetector(callback=self._on_hotword_detected)

    def process_text_input(self, text: str) -> Union[str, Any]:
        """Process text input"""
        if "llm" not in self.modules:
            self.logger.error("LLM module not initialized, cannot process text input")
            return "Sorry, my language model is not initialized, I cannot answer your question."

        llm = self.modules["llm"]
        text_config = config("llm.text") or {}
        model = text_config.get("model")

        try:
            if config("llm.parameters.stream"):
                return llm.generate_stream_response(model, text)
            else:
                return llm.generate_response(model, text)
        except Exception as e:
            self.logger.error(f"Error processing text input: {e}")
            return f"Error processing your request: {str(e)}"

    def process_image_input(self, image: str, text: str) -> Union[str, Any]:
        """Process image input with optional text description

        Args:
            image: Base64 encoded image or image URL
            text: Text description or question about the image

        Returns:
            Model response or error message
        """
        if "llm" not in self.modules:
            self.logger.error("LLM module not initialized, cannot process image input")
            return "Sorry, my language model is not initialized, I cannot analyze the image."

        llm = self.modules["llm"]
        vision_config = config("llm.vision") or {}

        if not vision_config.get("enable", False):
            self.logger.error("Vision model is disabled")
            return "Sorry, image analysis is not enabled."

        model = vision_config.get("model")

        try:
            if config("llm.parameters.stream"):
                return llm.generate_stream_response_with_image(model, text, image)
            else:
                return llm.generate_response_with_image(model, text, image)
        except Exception as e:
            self.logger.error(f"Error processing image input: {e}")
            return f"Error processing your request: {str(e)}"

    def _on_hotword_detected(self):
        """Hotword detection callback function"""
        self.logger.info("Hotword detected, starting to listen...")
        # 从配置中获取回应语，如果没有则使用默认值
        response_text = config("speech.hot_word.response_text") or "我在"
        self.speak(response_text)

        try:
            # 1. Check if we should capture image
            capture_image = False
            if "camera" in self.modules and config("llm.vision.enable"):
                # TODO: Add logic to determine when to capture image
                # For example: based on time of day, user preferences, or specific commands
                capture_image = False  # For now, default to False

            # 2. Recognize speech
            if config("speech.asr.stream.enable"):
                self.logger.info("Starting streaming speech recognition...")
                asr = self.modules["asr"]
                asr.start_streaming()
                try:
                    user_input = ""
                    while True:
                        result = asr.get_streaming_result(timeout=1.0)
                        if result:
                            user_input = result
                            break
                finally:
                    asr.stop_streaming()
            else:
                user_input = self.recognize_speech()

            if not user_input:
                self.logger.warning("No speech recognized or ASR failed.")
                # 提供反馈，告知用户没有听清
                no_speech_response = (
                    config("speech.asr.no_speech_response") or "抱歉，我没有听清楚"
                )
                self.speak(no_speech_response)
                return

            self.logger.info(f"Recognized speech: '{user_input}'")

            # 3. Process input (with or without image)
            if capture_image:
                image_base64 = self.modules["camera"].capture_frame_base64()
                response = self.process_image_input(image_base64, user_input)
            else:
                response = self.process_text_input(user_input)

            # 4. Speak the response
            if config("llm.parameters.stream"):
                full_response = ""
                self.logger.info("Assistant streaming response:")
                for chunk in response:
                    full_response += chunk
                    # Use TTS streaming if enabled
                    if config("speech.tts.stream.enable"):
                        self.stream_speak(chunk)

                self.logger.info(f"Final response: '{full_response}'")
                if full_response and not config("speech.tts.stream.enable"):
                    self.speak(full_response)
            else:
                self.logger.info(f"Assistant response: '{response}'")
                if response:
                    self.speak(response)

        except Exception as e:
            self.logger.error(f"Error in hotword callback: {e}", exc_info=True)
            self.speak("抱歉，我遇到了一些问题。")

    def capture_image(self):
        """Capture image"""
        if "camera" not in self.modules:
            self.logger.error("Camera module not initialized, cannot capture image")
            return None

        try:
            return self.modules["camera"].capture_frame()
        except Exception as e:
            self.logger.error(f"Error capturing image: {e}")
            return None

    def get_camera_info(self):
        """Get camera information and parameters"""
        if "camera" not in self.modules:
            self.logger.error("Camera module not initialized")
            return None

        try:
            return self.modules["camera"].get_camera_info()
        except Exception as e:
            self.logger.error(f"Error getting camera info: {e}")
            return None

    def stream_speak(self, text: str) -> bool:
        """Stream text to speech

        Args:
            text: Text chunk to speak

        Returns:
            Whether the text chunk was successfully processed
        """
        if "tts" not in self.modules:
            self.logger.error(
                "Speech synthesis module not initialized, cannot stream speech"
            )
            return False

        try:
            return self.modules["tts"].process_stream(text)
        except Exception as e:
            self.logger.error(f"Error streaming speech: {e}")
            return False

    def recognize_speech(self) -> str:
        """Recognize speech"""
        if "asr" not in self.modules:
            self.logger.error(
                "Speech recognition module not initialized, cannot recognize speech"
            )
            return ""

        try:
            return self.modules["asr"].recognize_from_microphone()
        except Exception as e:
            self.logger.error(f"Error recognizing speech: {e}")
            return ""

    def speak(self, text: str) -> bool:
        """Play speech"""
        if "tts" not in self.modules:
            self.logger.error(
                "Speech synthesis module not initialized, cannot play speech"
            )
            return False

        self.logger.info(f"Speaking: {text}")
        try:
            if config("speech.tts.stream.enable"):
                success = self.modules["tts"].process_stream(text)
                self.modules["tts"].flush_stream()
                return success
            else:
                return self.modules["tts"].speak(text)
        except Exception as e:
            self.logger.error(f"Error playing speech: {e}")
            return False

    def run(self):
        """Starts the assistant and waits for hotword detection."""
        self.logger.info(f"Starting {config('system.name')}. Press Ctrl+C to exit.")
        if "hotword" in self.modules and self.modules["hotword"]:
            try:
                self.modules["hotword"].start()
            except Exception as e:
                self.logger.error(f"Failed to start hotword detector: {e}")
                return

        # Keep the main thread alive to allow background threads to run.
        # Shutdown is handled by the signal handler in main.py.
        while True:
            time.sleep(1)

    def shutdown(self):
        """Shutdown assistant, release resources"""
        self.logger.info("Shutting down assistant...")

        # Stop hotword detection
        if "hotword" in self.modules:
            try:
                self.modules["hotword"].stop()
            except Exception as e:
                self.logger.error(f"Error stopping hotword detection: {e}")
