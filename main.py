import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import pyjokes
from TTS.api import TTS
import sounddevice as sd

# Load multi-speaker model
tts = TTS("tts_models/en/vctk/vits")


def speak(text: str = ""):
    print(f"Assistant: {text}")
    try:
        audio = tts.tts(text=text, speaker="p226")
        sample_rate = tts.synthesizer.output_sample_rate
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        print(e)
        print("Speech output not supported in Colab.")


def wish_user():
    hour = int(datetime.datetime.now().hour)
    if hour < 12:
        speak("Good Morning!")
    elif hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am your voice assistant. How can I help you today?")


def take_command():
    return input("You (type your command): ").lower()


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)

    try:
        # TODO: Use whisper model for offline recognition
        query = recognizer.recognize_google(
            audio,
            language="en-US",
        )
        print(f"You said: {query}\n")
    except sr.UnknownValueError:
        print("Could not understand audio, please say that again.")
        return "None"
    except sr.RequestError as e:
        print(f"Offline recognizer error: {e}. Ensure 'pocketsphinx' is installed.")
        return "None"
    return query.lower()


def run_assistant():
    wish_user()
    while True:
        # query = take_command()
        query = recognize_speech()

        if "wikipedia" in query:
            speak("Searching Wikipedia...")
            query = query.replace("wikipedia", "")
            try:
                result = wikipedia.summary(query, sentences=2)
                speak("According to Wikipedia:")
                speak(result)
            except:
                speak("Sorry, I couldn't find anything.")

        elif "open youtube" in query:
            speak("Opening YouTube...")
            webbrowser.open("https://www.youtube.com/")

        elif "open google" in query:
            speak("Opening Google...")
            webbrowser.open("https://www.google.com/")

        elif "time" in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"The current time is {strTime}")

        elif "joke" in query:
            joke = pyjokes.get_joke()
            speak(joke)

        elif "exit" in query or "bye" in query:
            speak("Goodbye! Have a nice day!")
            break

        else:
            speak("Sorry, I didn't understand that. Try again.")


def main():
    print("Hello from voice-assistant!")
    run_assistant()


if __name__ == "__main__":
    main()
