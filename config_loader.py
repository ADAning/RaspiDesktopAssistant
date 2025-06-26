import os
import yaml
from dotenv import load_dotenv


class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Load environment variables from .env file
        load_dotenv()

        # Determine the configuration file path
        config_path = os.getenv("CONFIG", "./config.yaml")

        try:
            with open(config_path, "r") as file:
                self._config = yaml.safe_load(file)

            # Inject sensitive information from environment variables
            if "llm" in self._config and "text" in self._config["llm"]:
                api_key = os.getenv(
                    "LLM_API_KEY", self._config["llm"]["text"].get("api_key")
                )
                if api_key:
                    self._config["llm"]["text"]["api_key"] = api_key

            # Inject vision API key if available
            if (
                "llm" in self._config
                and "vision" in self._config["llm"]
                and self._config["llm"]["vision"].get("enable")
            ):
                vision_api_key = os.getenv(
                    "LLM_VISION_API_KEY", self._config["llm"]["vision"].get("api_key")
                )
                if vision_api_key:
                    self._config["llm"]["vision"]["api_key"] = vision_api_key

            # Additional environment variable injections can be added here
            # ...

        except FileNotFoundError:
            raise Exception(f"Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML configuration: {e}")

    def get(self, path=None, default=None):
        """
        Get configuration values, supports dot-separated paths.
        Example: get('llm.cloud_api.key')
        """
        if self._config is None:
            raise Exception("Configuration not loaded")

        if path is None:
            return self._config

        keys = path.split(".")
        data = self._config
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return default
        return data

    def update(self, path, value):
        """
        Update configuration values, supports dot-separated paths.
        """
        if self._config is None:
            raise Exception("Configuration not loaded")

        keys = path.split(".")
        data = self._config
        for i, key in enumerate(keys[:-1]):
            if key not in data:
                data[key] = {}
            data = data[key]

        data[keys[-1]] = value

    def save_config(self, path=None):
        """
        Save the current configuration to a file.
        """
        config_path = path or os.getenv("ROBOT_CONFIG", "./config.yaml")
        with open(config_path, "w") as file:
            yaml.safe_dump(self._config, file, default_flow_style=False)


# Global configuration accessor
config = ConfigLoader().get
