import argparse
import sys
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from typing import Optional

import numpy as np
import pyjokes
import sounddevice as sd
import speech_recognition as sr
import torch
import webbrowser
import wikipedia
import whisper
from TTS.api import TTS

# Constants
SAMPLE_RATE = 16000
IS_LINUX = sys.platform.startswith("linux")


class VoiceAssistant:
    """Main voice assistant class handling TTS and command processing."""

    def __init__(self):
        self.tts = TTS("tts_models/en/vctk/vits")

    def speak(self, text: str = "") -> None:
        """Convert text to speech and play it."""
        if not text:
            return
        print(f"Assistant: {text}")
        try:
            audio = self.tts.tts(text=text, speaker="p226")
            sample_rate = self.tts.synthesizer.output_sample_rate
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            print(f"TTS Error: {e}")

    def wish_user(self) -> None:
        """Greet the user based on time of day."""
        hour = datetime.now().hour
        if hour < 12:
            self.speak("Good Morning!")
        elif hour < 18:
            self.speak("Good Afternoon!")
        else:
            self.speak("Good Evening!")
        self.speak("I am your voice assistant. How can I help you today?")

    def handle_command(self, query: str) -> bool:
        """
        Process user commands.
        Returns True if assistant should continue, False if should exit.
        """
        query = query.strip().lower()
        if not query or query == "none":
            return True

        if "wikipedia" in query:
            self._handle_wikipedia(query)
        elif "open youtube" in query:
            self.speak("Opening YouTube...")
            webbrowser.open("https://www.youtube.com/")
        elif "open google" in query:
            self.speak("Opening Google...")
            webbrowser.open("https://www.google.com/")
        elif "time" in query:
            current_time = datetime.now().strftime("%H:%M:%S")
            self.speak(f"The current time is {current_time}")
        elif "joke" in query:
            joke = pyjokes.get_joke()
            self.speak(joke)
        elif "exit" in query or "bye" in query:
            self.speak("Goodbye! Have a nice day!")
            return False
        else:
            self.speak("Sorry, I didn't understand that. Try again.")
        return True

    def _handle_wikipedia(self, query: str) -> None:
        """Handle Wikipedia search queries."""
        self.speak("Searching Wikipedia...")
        search_query = query.replace("wikipedia", "").strip()
        if not search_query:
            self.speak("Please specify what to search on Wikipedia.")
            return
        try:
            result = wikipedia.summary(search_query, sentences=2)
            self.speak("According to Wikipedia:")
            self.speak(result)
        except wikipedia.exceptions.DisambiguationError as e:
            self.speak(f"Multiple results found. {e.options[0]}")
        except wikipedia.exceptions.PageError:
            self.speak("Sorry, I couldn't find anything on Wikipedia.")
        except Exception as e:
            print(f"Wikipedia error: {e}")
            self.speak("Sorry, I couldn't find anything.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Voice Assistant with Whisper")
    parser.add_argument(
        "--model",
        default="medium",
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
        default=2.0,
        help="How real-time the recording is in seconds",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=3.0,
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


class WhisperTranscriber:
    """Handles Whisper-based speech transcription."""

    def __init__(
        self,
        model_name: str,
        non_english: bool,
        recorder: sr.Recognizer,
        source: sr.Microphone,
        record_timeout: float,
        phrase_timeout: float,
        data_queue: Queue,
    ):
        self.model_name = model_name
        self.non_english = non_english
        self.recorder = recorder
        self.source = source
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.data_queue = data_queue
        self.audio_model = self._load_model()
        self.phrase_time: Optional[datetime] = None
        self.phrase_bytes = bytes()
        self.transcription = [""]

    def _load_model(self) -> whisper.Whisper:
        """Load Whisper model."""
        model = self.model_name
        if model != "large" and not self.non_english:
            model = f"{model}.en"
        print(f"Loading Whisper model: {model}...")
        return whisper.load_model(model)

    def _record_callback(self, _: sr.Recognizer, audio: sr.AudioData) -> None:
        """Threaded callback to receive audio data."""
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def start_listening(self) -> None:
        """Start background audio listening."""
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
        self.recorder.listen_in_background(
            self.source,
            self._record_callback,
            phrase_time_limit=self.record_timeout,
        )
        print("Model loaded. Listening...\n")

    def process_audio_queue(self) -> Optional[str]:
        """
        Process audio from queue and return transcribed text if phrase is complete.
        Returns None if no new complete phrase, or empty string if processing incomplete phrase.
        """
        if self.data_queue.empty():
            return None

        now = datetime.now()
        phrase_complete = False

        # Check if phrase is complete (enough time passed)
        if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
            self.phrase_bytes = bytes()
            phrase_complete = True

        self.phrase_time = now

        # Combine audio data from queue
        audio_data = b"".join(list(self.data_queue.queue))
        self.data_queue.queue.clear()
        self.phrase_bytes += audio_data

        if not self.phrase_bytes:
            return None

        # Convert audio to numpy array
        audio_np = np.frombuffer(self.phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe
        try:
            result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            text = result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

        # Update transcription history
        if phrase_complete:
            self.transcription.append(text)
            print(f"You: {text}\n")
            return text
        else:
            self.transcription[-1] = text
            return ""


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
                sleep(0.25)
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
