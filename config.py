import argparse
import sys

# Constants
SAMPLE_RATE = 16000
IS_LINUX = sys.platform.startswith("linux")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Voice Assistant with Whisper")
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model to use",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    parser.add_argument(
        "--non_english",
        action="store_true",
        help="Don't use the English model",
    )
    parser.add_argument(
        "--energy_threshold",
        default=1000,
        help="Energy level for mic to detect",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=1.0,
        help="How real-time the recording is in seconds",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=1.5,
        help="How much empty space between recordings before considering it a new phrase",
        type=float,
    )
    if IS_LINUX:
        parser.add_argument(
            "--default_microphone",
            default="pulse",
            help="Default microphone name. Use 'list' to view available microphones",
            type=str,
        )
    return parser.parse_args()
