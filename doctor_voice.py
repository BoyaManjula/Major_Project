import pyttsx3
import os

def text_to_speech(text, output_path="response.wav"):
    try:
        if os.path.exists(output_path):
            os.remove(output_path)

        engine = pyttsx3.init()
        engine.save_to_file(text, output_path)
        engine.runAndWait()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path

        return None

    except Exception as e:
        print("TTS Error:", e)
        return None
