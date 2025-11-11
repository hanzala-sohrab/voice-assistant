from datetime import datetime
import webbrowser

import pyjokes
import sounddevice as sd
import wikipedia
from TTS.api import TTS


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
