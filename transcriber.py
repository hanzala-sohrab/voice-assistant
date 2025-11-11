from datetime import datetime, timedelta
from queue import Queue
from typing import Optional

import numpy as np
import speech_recognition as sr
import torch
import whisper


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
