# Voice Assistant

A Python-based voice assistant that uses OpenAI Whisper for speech recognition and TTS for text-to-speech responses. The assistant can perform various tasks through voice commands including web searches, opening websites, telling jokes, and more.

## Features

- 🎤 **Speech Recognition**: Uses OpenAI Whisper for accurate speech-to-text conversion
- 🔊 **Text-to-Speech**: Converts responses to natural-sounding speech
- 🌐 **Web Integration**: Open YouTube, Google, and search Wikipedia
- ⏰ **Time Queries**: Get the current time
- 😄 **Entertainment**: Tells jokes on command
- 🔧 **Configurable**: Customize microphone, model size, and detection thresholds

## Requirements

- Python 3.11 or higher
- Microphone for voice input
- Linux, macOS, or Windows

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd voice-assistant
```

2. Install dependencies using `uv` (recommended) or `pip`:
```bash
# Using uv
uv sync

# Or using pip
pip install -r requirements.txt
```

## Usage

Run the voice assistant:

```bash
uv run main.py
```

### Command-Line Arguments

- `--model`: Whisper model to use (default: `medium`)
  - Options: `tiny`, `base`, `small`, `medium`, `large`
  - Larger models are more accurate but slower

- `--non_english`: Use multilingual model instead of English-only

- `--energy_threshold`: Energy level for microphone to detect (default: 1000)
  - Lower values = more sensitive to quiet sounds
  - Higher values = less sensitive, reduces background noise

- `--record_timeout`: How real-time the recording is in seconds (default: 2.0)

- `--phrase_timeout`: How much empty space between recordings before considering it a new phrase in seconds (default: 3.0)

- `--default_microphone`: Default microphone name (Linux only, default: `pulse`)
  - Use `--default_microphone list` to view available microphones

### Examples

Run with a larger, more accurate model:
```bash
uv run main.py --model large
```

Use a faster, smaller model:
```bash
uv run main.py --model tiny
```

List available microphones (Linux):
```bash
uv run main.py --default_microphone list
```

## Supported Commands

Once the assistant is running, you can use the following voice commands:

- **"Wikipedia [topic]"** - Search Wikipedia for information about a topic
- **"Open YouTube"** - Opens YouTube in your default browser
- **"Open Google"** - Opens Google in your default browser
- **"Time"** - Get the current time
- **"Joke"** - Hear a random joke
- **"Exit"** or **"Bye"** - Exit the assistant

## How It Works

1. The assistant starts by greeting you based on the time of day
2. It continuously listens to your microphone using Whisper for transcription
3. When you finish speaking (after a phrase timeout), it processes your command
4. The assistant responds with both printed text and spoken audio

## Troubleshooting

### Microphone Not Detected
- On Linux, use `--default_microphone list` to see available devices
- Ensure your microphone permissions are granted
- Check that PyAudio is properly installed

### Low Accuracy
- Try using a larger model: `--model large`
- Adjust `--energy_threshold` to better detect your voice
- Ensure a quiet environment for better transcription

### Performance Issues
- Use a smaller model (`tiny` or `base`) for faster processing
- Reduce `--record_timeout` and `--phrase_timeout` for quicker responses

## License

MIT License - see [LICENSE](LICENSE) file for details.
