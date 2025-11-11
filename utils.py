import sys
from typing import Optional

import speech_recognition as sr

from config import SAMPLE_RATE, IS_LINUX


def setup_microphone(mic_name: Optional[str] = None) -> sr.Microphone:
    """Setup and return microphone source."""
    if not IS_LINUX:
        return sr.Microphone(sample_rate=SAMPLE_RATE)

    if not mic_name or mic_name == "list":
        print("Available microphone devices:")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f'  [{index}] "{name}"')
        sys.exit(0)

    # Find microphone by name
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        if mic_name.lower() in name.lower():
            return sr.Microphone(sample_rate=SAMPLE_RATE, device_index=index)

    print(f"Warning: Microphone '{mic_name}' not found. Using default.")
    return sr.Microphone(sample_rate=SAMPLE_RATE)
