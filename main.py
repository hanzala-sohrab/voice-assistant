from queue import Queue
from time import sleep

import speech_recognition as sr

from assistant import VoiceAssistant
from config import IS_LINUX, parse_arguments
from transcriber import WhisperTranscriber
from utils import setup_microphone


def run_assistant() -> None:
    """Main assistant loop."""
    args = parse_arguments()

    # Initialize assistant
    assistant = VoiceAssistant()
    assistant.wish_user()

    # Setup microphone
    mic_name = getattr(args, "default_microphone", None) if IS_LINUX else None
    source = setup_microphone(mic_name)

    # Setup speech recognizer
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Initialize transcriber
    data_queue = Queue()
    transcriber = WhisperTranscriber(
        model_name=args.model,
        non_english=args.non_english,
        recorder=recorder,
        source=source,
        record_timeout=args.record_timeout,
        phrase_timeout=args.phrase_timeout,
        data_queue=data_queue,
    )
    transcriber.start_listening()

    # Main loop
    try:
        while True:
            # Process audio queue
            query = transcriber.process_audio_queue()

            # Only process commands when we have a complete phrase
            if query and query != "none":
                if not assistant.handle_command(query):
                    break

            # Reduce CPU usage when queue is empty
            if query is None:
                sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        assistant.speak("Goodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise


def main() -> None:
    """Entry point for the voice assistant."""
    print("Voice Assistant starting...")
    run_assistant()


if __name__ == "__main__":
    main()
