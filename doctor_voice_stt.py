import os
from groq import Groq

def transcribe_with_groq(audio_path):
    api_key = os.getenv("GROQ_API_KEY")

    # If no API key, return quickly
    if not api_key:
        return "Speech transcription unavailable (no API key)."

    try:
        client = Groq(api_key=api_key)

        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f
            )

        return transcription.text

    except Exception as e:
        print("STT Error:", e)
        return "Speech transcription unavailable (offline or blocked internet)."
