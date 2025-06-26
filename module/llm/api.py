"""
For the Raspberry Pi desktop assistant's large language model API.
"""

import types
from functools import wraps
from openai import OpenAI
from typing import Union, List, Dict, Optional
from config_loader import config
import logging

logger = logging.getLogger(__name__)


class LLMConfigError(Exception):
    """Exception raised for LLM configuration errors."""

    pass


class LLMVisionDisabledError(Exception):
    """Exception raised when trying to use vision features while they are disabled."""

    pass


class LLM:
    def __init__(
        self,
        text_api_key=None,
        text_base_url=None,
        vision_api_key=None,
        vision_base_url=None,
        system_prompt=None,
    ):
        """
        Initialize the LLM API client.
        :param text_api_key: API key for text model
        :param text_base_url: Base URL for text model API
        :param vision_api_key: API key for vision model
        :param vision_base_url: Base URL for vision model API
        :param system_prompt: System prompt for the assistant
        :raises LLMConfigError: If text model configuration is invalid
        """
        # Load config
        llm_config = config.get("llm", {})
        text_config = llm_config.get("text", {})
        vision_config = llm_config.get("vision", {})

        # Text model configuration (required)
        self.text_api_key = text_api_key or text_config.get("api_key")
        self.text_base_url = text_base_url or text_config.get("base_url")
        self.text_model = text_config.get("model", "deepseek-chat")

        if not self.text_api_key or not self.text_base_url:
            raise LLMConfigError(
                "Text model configuration is required but api_key or base_url is missing"
            )

        text_param_config = text_config.get("parameters", {})
        self.text_parameters = {
            "max_tokens": text_param_config.get("max_tokens", 4096),
            "temperature": text_param_config.get("temperature", 1.0),
            "top_p": text_param_config.get("top_p", 1),
            "frequency_penalty": text_param_config.get("frequency_penalty", 0.0),
            "presence_penalty": text_param_config.get("presence_penalty", 0.0),
            "stream": text_param_config.get("stream", True),
        }

        # Vision model configuration (optional)
        self.vision_enabled = vision_config.get("enable", False)
        self.vision_client = None

        if self.vision_enabled:
            self.vision_api_key = vision_api_key or vision_config.get("api_key")
            self.vision_base_url = vision_base_url or vision_config.get("base_url")

            if not self.vision_api_key or not self.vision_base_url:
                logger.warning(
                    "Vision is enabled but api_key or base_url is missing, disabling vision features"
                )
                self.vision_enabled = False
            else:
                self.vision_model = vision_config.get("model", "qwen-vl")
                vision_param_config = vision_config.get("parameters", {})
                self.vision_parameters = {
                    "max_tokens": vision_param_config.get("max_tokens", 4096),
                    "temperature": vision_param_config.get("temperature", 1.0),
                    "top_p": vision_param_config.get("top_p", 1),
                    "frequency_penalty": vision_param_config.get(
                        "frequency_penalty", 0.0
                    ),
                    "presence_penalty": vision_param_config.get(
                        "presence_penalty", 0.0
                    ),
                    "stream": vision_param_config.get("stream", True),
                }

                # Initialize vision client
                self.vision_client = OpenAI(
                    api_key=self.vision_api_key, base_url=self.vision_base_url
                )
        else:
            logger.info("Vision features are disabled")

        # Initialize text client (required)
        self.text_client = OpenAI(
            api_key=self.text_api_key, base_url=self.text_base_url
        )

        # Initialize shared conversation history
        self.max_turns = text_param_config.get("max_turns", 10)
        self.system_prompt = {
            "role": "system",
            "content": system_prompt
            or text_config.get("init_prompt", "You are a helpful assistant."),
        }
        self.messages = [self.system_prompt]

    def add_message(self, message):
        """
        Add a message to the conversation history
        :param message: Message to add
        """
        self.messages.append(message)
        while len(self.messages) > self.max_turns:
            if self.messages[1]["role"] != "system":
                del self.messages[1]
            else:
                if len(self.messages) > 2:
                    del self.messages[2]

    def create_image_content(
        self, image: Union[str, List[str]], text: Optional[str] = None
    ) -> List[Dict]:
        """
        Create image content for the LLM
        :param image: Image input content
        :param text: Optional text description
        :return: Formatted message content list
        :raises LLMVisionDisabledError: If vision features are disabled
        """
        if not self.vision_enabled:
            raise LLMVisionDisabledError(
                "Cannot create image content: Vision features are disabled"
            )

        content = []

        # add text part (if any)
        if text:
            content.append({"type": "text", "text": text})

        # process single image or image list
        images = image if isinstance(image, list) else [image]
        for img in images:
            # if it is a URL, use it directly; if it is a base64, add prefix
            if not img.startswith(("http://", "https://")):
                img = f"data:image/jpeg;base64,{img}"

            content.append({"type": "image_url", "image_url": {"url": img}})

        return content

    def get_messages(self):
        """Obtain the current conversation history."""
        return self.messages

    def clear_messages(self, keep_system=True):
        """
        Clear the conversation history.
        :param keep_system: Whether to retain the system prompt in the history
        """
        if keep_system:
            self.messages = [self.system_prompt]
        else:
            self.messages = []

    def with_message_history(func):
        """
        Decorator to handle message history and non-streaming/streaming responses.
        """

        @wraps(func)
        def wrapper(self, model, user_input, *args, **kwargs):
            # 1. add user input to message history
            self.add_message({"role": "user", "content": user_input})

            # 2. call the wrapped function to get the response
            result = func(self, model, user_input, *args, **kwargs)

            # 3. process the result
            if isinstance(result, types.GeneratorType):
                # streaming response
                def stream_wrapper():
                    full_response = ""
                    try:
                        for content in result:
                            full_response += content
                            yield content
                    finally:
                        # 4. add the full response to message history
                        if full_response:
                            self.add_message(
                                {"role": "assistant", "content": full_response}
                            )

                return stream_wrapper()
            else:
                # non-streaming response
                # non-streaming functions should return (content, message)
                if isinstance(result, tuple) and len(result) == 2:
                    content, message = result
                    self.add_message(message)
                    return content
                else:
                    raise ValueError(
                        "Non-streaming function must return a tuple (content, message)"
                    )

        return wrapper

    @with_message_history
    def generate_response(self, model: Optional[str], user_input: str):
        """
        Obtain a response from the LLM using text model
        :param model: Optional model name to override default text model
        :param user_input: User input content
        :return: LLM response content and the message object
        """
        response = self.text_client.chat.completions.create(
            model=model or self.text_model,
            messages=self.messages,
            stream=False,
            **self.text_parameters,
        )
        message = response.choices[0].message

        return message.content, message

    @with_message_history
    def generate_stream_response(self, model: Optional[str], user_input: str):
        """
        Obtain a streaming response from the LLM using text model
        :param model: Optional model name to override default text model
        :param user_input: User input content
        :return: A generator yielding chunks of the LLM response content
        """
        stream = self.text_client.chat.completions.create(
            model=model or self.text_model,
            messages=self.messages,
            stream=True,
            **self.text_parameters,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield content

    @with_message_history
    def generate_response_with_image(
        self, model: Optional[str], user_input: str, image: Union[str, List[str]]
    ):
        """
        Obtain a response from the LLM with image input using vision model
        :param model: Optional model name to override default vision model
        :param user_input: User input content
        :param image: Image input content (base64 string or URL)
        :return: LLM response content and the message object
        :raises LLMVisionDisabledError: If vision features are disabled
        """
        if not self.vision_enabled:
            raise LLMVisionDisabledError(
                "Cannot generate response with image: Vision features are disabled"
            )

        # create image content
        content = self.create_image_content(image, user_input)

        # replace the last user message content
        self.messages[-1]["content"] = content

        response = self.vision_client.chat.completions.create(
            model=model or self.vision_model,
            messages=self.messages,
            stream=False,
            **self.vision_parameters,
        )
        message = response.choices[0].message

        return message.content, message

    @with_message_history
    def generate_stream_response_with_image(
        self, model: Optional[str], user_input: str, image: Union[str, List[str]]
    ):
        """
        Obtain a streaming response from the LLM with image input using vision model
        :param model: Optional model name to override default vision model
        :param user_input: User input content
        :param image: Image input content (base64 string or URL)
        :return: A generator yielding chunks of the LLM response content
        :raises LLMVisionDisabledError: If vision features are disabled
        """
        if not self.vision_enabled:
            raise LLMVisionDisabledError(
                "Cannot generate streaming response with image: Vision features are disabled"
            )

        # create image content
        content = self.create_image_content(image, user_input)

        # replace the last user message content
        self.messages[-1]["content"] = content

        stream = self.vision_client.chat.completions.create(
            model=model or self.vision_model,
            messages=self.messages,
            stream=True,
            **self.vision_parameters,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield content

    def get_model_list(self):
        """Get list of available models."""
        models = {"text": self.text_client.models.list()}
        if self.vision_enabled:
            models["vision"] = self.vision_client.models.list()
        return models
