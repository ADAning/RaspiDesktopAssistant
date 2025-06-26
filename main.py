import os
import sys
import time
import signal
import logging
import argparse
from typing import Optional

from core import Assistant
from config_loader import config, ConfigLoader

# Global variables
assistant: Optional[Assistant] = None


def signal_handler(sig, frame):
    """Handling signals, gracefully exit the program"""
    print("\nShutting down assistant...")
    sys.exit(0)


def parse_arguments():
    """Parsing command line arguments"""
    parser = argparse.ArgumentParser(description="Raspberry Pi Desktop Assistant")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    return parser.parse_args()


def setup_environment(args):
    """Setting environment variables"""
    if args.config:
        os.environ["CONFIG"] = args.config
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level


def main():
    """Main function"""
    global assistant

    # Parsing command line arguments
    args = parse_arguments()

    # Setting environment variables
    setup_environment(args)

    # Registering signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initializing assistant
        assistant = Assistant()

        # Start assistant's main loop
        assistant.run()

    except Exception as e:
        logging.critical(f"Error initializing or running assistant: {e}", exc_info=True)
    finally:
        if assistant:
            assistant.shutdown()
        print("Assistant closed")


if __name__ == "__main__":
    main()
