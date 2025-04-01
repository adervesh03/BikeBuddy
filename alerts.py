# alerts.py
import os
import sys
import time
import subprocess
import tempfile
from gtts import gTTS

# Generate and play speech from text
def speak_text(text):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            temp_filename = tmp_file.name

        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_filename)

        try:
            subprocess.call(['mpg123', '-q', temp_filename])
        except FileNotFoundError:
            try:
                subprocess.call(['aplay', temp_filename])
            except FileNotFoundError:
                if sys.platform == "darwin":
                    subprocess.call(['afplay', temp_filename])
                elif sys.platform == "win32":
                    os.startfile(temp_filename)

        time.sleep(1)
        os.remove(temp_filename)
    except Exception as e:
        print(f"Text-to-speech error: {e}")
