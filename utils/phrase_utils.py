"""
phrase_utils.py — Gesture sequence → word / common phrase mapping.
Text-to-speech via pyttsx3.
"""

import threading

# ── Common ASL phrase dictionary ──────────────────────────────────────────────
# Key: tuple of UPPERCASE letters that spell the phrase
# Value: (phrase_display, category)

PHRASE_DICT: dict[str, tuple[str, str]] = {
    "HELLO"         : ("Hello! 👋",                 "Greeting"),
    "HI"            : ("Hi! 😊",                    "Greeting"),
    "BYE"           : ("Goodbye! 👋",               "Farewell"),
    "GOODBYE"       : ("Goodbye! 👋",               "Farewell"),
    "GOODMORNING"   : ("Good Morning! 🌅",          "Greeting"),
    "GOODNIGHT"     : ("Good Night! 🌙",            "Farewell"),
    "GOODAFTERNOON" : ("Good Afternoon! ☀️",        "Greeting"),
    "THANKYOU"      : ("Thank You! 🙏",             "Courtesy"),
    "THANKS"        : ("Thanks! 🙏",                "Courtesy"),
    "SORRY"         : ("I'm Sorry! 😔",             "Courtesy"),
    "PLEASE"        : ("Please 🙏",                 "Courtesy"),
    "YES"           : ("Yes ✅",                    "Response"),
    "NO"            : ("No ❌",                     "Response"),
    "HELP"          : ("Help! 🆘",                  "Emergency"),
    "ILOVEYOU"      : ("I Love You ❤️",             "Emotion"),
    "OK"            : ("OK! 👍",                    "Response"),
    "NICE"          : ("Nice! 😊",                  "Emotion"),
    "GREAT"         : ("Great! 🎉",                 "Emotion"),
    "WATER"         : ("Water 💧",                  "Need"),
    "FOOD"          : ("Food 🍽️",                   "Need"),
    "BATHROOM"      : ("Bathroom 🚻",              "Need"),
    "PAIN"          : ("Pain / I'm hurting 😣",    "Emergency"),
    "CALL"          : ("Call Someone 📞",          "Emergency"),
    "STOP"          : ("Stop ✋",                   "Command"),
    "WAIT"          : ("Wait ⏳",                   "Command"),
    "COME"          : ("Come Here 👈",              "Command"),
    "GO"            : ("Go / Let's Go 🚶",          "Command"),
    "NAME"          : ("What is your name? 🤔",    "Question"),
    "WHERE"         : ("Where? 📍",                "Question"),
    "WHEN"          : ("When? 🕐",                  "Question"),
    "WHY"           : ("Why? 🤔",                   "Question"),
    "HOW"           : ("How? 🤔",                   "Question"),
    "WHAT"          : ("What? 🤔",                  "Question"),
}


def lookup_phrase(letters: str) -> tuple[str | None, str | None]:
    """
    Strip spaces from letters and look up in PHRASE_DICT.
    Returns (phrase_display, category) or (None, None).
    """
    key = letters.upper().replace(" ", "")
    result = PHRASE_DICT.get(key)
    if result:
        return result
    return None, None


def get_all_phrases() -> list[dict]:
    """Return list of phrase dicts for display."""
    return [
        {"key": k, "phrase": v[0], "category": v[1]}
        for k, v in PHRASE_DICT.items()
    ]


def speak_text(text: str, lang_code: str = "en"):
    """
    Speak text via gTTS + pygame in a background thread.
    Works for all languages supported by Google TTS.
    
    Args:
        text: The text to speak (can be in any language)
        lang_code: gTTS language code (e.g. 'en', 'hi', 'mr', 'ta')
    """
    def _speak():
        try:
            import os, tempfile
            from gtts import gTTS
            import pygame

            tts = gTTS(text=text, lang=lang_code, slow=False)

            # Save to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name
                tts.save(tmp_path)

            # Play using pygame
            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()

            # Wait for playback to finish then clean up
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.music.unload()
            os.remove(tmp_path)

        except ImportError:
            # Fallback to pyttsx3 if gTTS/pygame not available
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
        except Exception:
            pass  # Silently fail if TTS unavailable

    threading.Thread(target=_speak, daemon=True).start()


# ── Letter accumulation helper ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70
LETTER_HOLD_FRAMES   = 1         # set to 1 for instant capture via camera_input snapshots
CLEAR_PHRASE_FRAMES  = 40       # frames of no hand → clear phrase buffer


class GestureWordBuilder:
    """Stateful letter-by-letter word/phrase builder."""

    def __init__(self):
        self.current_letter  : str  = ""
        self.hold_count      : int  = 0
        self.letters         : str  = ""
        self.no_hand_count   : int  = 0
        self.last_added      : str  = ""

    def update(self, label: str | None, confidence: float) -> dict:
        """
        Feed a prediction into the builder.

        Returns:
            dict with keys: new_letter_added, letters, phrase_detected, phrase_display
        """
        result = {
            "new_letter_added": False,
            "letters"         : self.letters,
            "phrase_detected" : None,
            "phrase_display"  : None,
        }

        if label is None:
            self.no_hand_count += 1
            if self.no_hand_count >= CLEAR_PHRASE_FRAMES:
                self.letters       = ""
                self.last_added    = ""
                self.hold_count    = 0
                self.current_letter = ""
                self.no_hand_count  = 0
            self.current_letter = ""
            result["letters"] = self.letters
            return result

        self.no_hand_count = 0

        if confidence >= CONFIDENCE_THRESHOLD:
            if label == self.current_letter:
                self.hold_count += 1
            else:
                self.current_letter = label
                self.hold_count     = 1
            self.last_added = ""  # Allow same letter again after hold cycle reset

            if self.hold_count == LETTER_HOLD_FRAMES:
                self.letters += label
                result["new_letter_added"] = True
                result["letters"]          = self.letters
                self.hold_count            = 0  # reset for next letter

                # Check for phrase
                phrase, category = lookup_phrase(self.letters)
                if phrase:
                    result["phrase_detected"] = category
                    result["phrase_display"]  = phrase

        return result

    def clear(self):
        self.letters        = ""
        self.hold_count     = 0
        self.current_letter = ""
        self.last_added     = ""
        self.no_hand_count  = 0

    def backspace(self):
        if self.letters:
            self.letters = self.letters[:-1]
