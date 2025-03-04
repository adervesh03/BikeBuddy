from gtts import gTTS
import os

def speak(text, lang='en', tld='com'):
    """
    Speak text with specified language and accent
    lang: Language code (e.g., 'en', 'fr', 'es')
    tld: Top-level domain for different accents ('com', 'co.uk', 'com.au', 'co.in')
    """
    tts = gTTS(text=text, lang=lang, tld=tld)
    tts.save("speak.mp3")
    os.system("mpg321 speak.mp3")
    os.remove("speak.mp3")

# Examples with different voices:
speak("Hello, I'm using American English", tld='com')  # US voice
speak("Hello, I'm using British English", tld='co.uk')  # UK voice
speak("Hello, I'm using Indian English", tld='co.in')  # Indian voice
speak("Hello, I'm using Australian English", tld='com.au')  # Australian voice

# Different languages:
speak("Bonjour, je parle français", lang='fr')  # French
speak("Hola, hablo español", lang='es')  # Spanish