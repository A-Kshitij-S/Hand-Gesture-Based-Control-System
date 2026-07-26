"""
translation_utils.py — Translate recognized ASL text to multiple languages.
Uses deep-translator (Google backend, no API key needed).
"""

from deep_translator import GoogleTranslator

# ── All supported translation languages ──────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "English"            : "en",
    # ── Indian Regional Languages ──
    "Hindi"              : "hi",
    "Bengali"            : "bn",
    "Telugu"             : "te",
    "Marathi"            : "mr",
    "Tamil"              : "ta",
    "Gujarati"           : "gu",
    "Kannada"            : "kn",
    "Malayalam"          : "ml",
    "Punjabi"            : "pa",
    "Odia"               : "or",
    "Urdu"               : "ur",
    "Assamese"           : "as",
    "Nepali"             : "ne",
    "Sindhi"             : "sd",
    "Konkani"            : "gom",
    "Sanskrit"           : "sa",
    "Maithili"           : "mai",
    "Kashmiri"           : "ks",
    "Dogri"              : "doi",
    "Manipuri (Meitei)"  : "mni-Mtei",
    # ── International Languages ──
    "Spanish"            : "es",
    "French"             : "fr",
    "Arabic"             : "ar",
    "Chinese (Simplified)": "zh-CN",
    "Japanese"           : "ja",
    "German"             : "de",
    "Portuguese"         : "pt",
    "Russian"            : "ru",
    "Korean"             : "ko",
    "Italian"            : "it",
    "Turkish"            : "tr",
    "Dutch"              : "nl",
    "Indonesian"         : "id",
    "Thai"               : "th",
    "Vietnamese"         : "vi",
    "Swahili"            : "sw",
}

# gTTS language codes for speech (only languages gTTS supports)
# Maps display name → gTTS lang code
GTTS_LANGUAGES = {
    "English"            : "en",
    "Hindi"              : "hi",
    "Bengali"            : "bn",
    "Telugu"             : "te",
    "Marathi"            : "mr",
    "Tamil"              : "ta",
    "Gujarati"           : "gu",
    "Kannada"            : "kn",
    "Malayalam"          : "ml",
    "Punjabi"            : "pa",
    "Nepali"             : "ne",
    "Urdu"               : "ur",
    "Spanish"            : "es",
    "French"             : "fr",
    "Arabic"             : "ar",
    "Chinese (Simplified)": "zh-CN",
    "Japanese"           : "ja",
    "German"             : "de",
    "Portuguese"         : "pt",
    "Russian"            : "ru",
    "Korean"             : "ko",
    "Italian"            : "it",
    "Turkish"            : "tr",
    "Dutch"              : "nl",
    "Indonesian"         : "id",
    "Thai"               : "th",
    "Vietnamese"         : "vi",
}

LANGUAGE_FLAGS = {
    "English"            : "🇬🇧",
    "Hindi"              : "🇮🇳",
    "Bengali"            : "🇮🇳",
    "Telugu"             : "🇮🇳",
    "Marathi"            : "🇮🇳",
    "Tamil"              : "🇮🇳",
    "Gujarati"           : "🇮🇳",
    "Kannada"            : "🇮🇳",
    "Malayalam"          : "🇮🇳",
    "Punjabi"            : "🇮🇳",
    "Odia"               : "🇮🇳",
    "Urdu"               : "🇮🇳",
    "Assamese"           : "🇮🇳",
    "Nepali"             : "🇮🇳",
    "Sindhi"             : "🇮🇳",
    "Konkani"            : "🇮🇳",
    "Sanskrit"           : "🇮🇳",
    "Maithili"           : "🇮🇳",
    "Kashmiri"           : "🇮🇳",
    "Dogri"              : "🇮🇳",
    "Manipuri (Meitei)"  : "🇮🇳",
    "Spanish"            : "🇪🇸",
    "French"             : "🇫🇷",
    "Arabic"             : "🇸🇦",
    "Chinese (Simplified)": "🇨🇳",
    "Japanese"           : "🇯🇵",
    "German"             : "🇩🇪",
    "Portuguese"         : "🇧🇷",
    "Russian"            : "🇷🇺",
    "Korean"             : "🇰🇷",
    "Italian"            : "🇮🇹",
    "Turkish"            : "🇹🇷",
    "Dutch"              : "🇳🇱",
    "Indonesian"         : "🇮🇩",
    "Thai"               : "🇹🇭",
    "Vietnamese"         : "🇻🇳",
    "Swahili"            : "🇰🇪",
}


def translate_text(text: str, target_lang_code: str) -> str:
    """
    Translate text from English to target language.

    Args:
        text: English input text
        target_lang_code: BCP-47 language code e.g. 'hi', 'fr'

    Returns:
        Translated string or error message.
    """
    if not text or not text.strip():
        return ""
    try:
        result = GoogleTranslator(source='en', target=target_lang_code).translate(text.strip())
        return result or text
    except Exception as e:
        return f"[Translation error: {e}]"


def translate_to_languages(text: str, lang_names: list[str]) -> dict[str, str]:
    """
    Translate text to multiple languages at once.

    Args:
        text: English text
        lang_names: list of language display names from SUPPORTED_LANGUAGES

    Returns:
        dict mapping language name → translated text
    """
    results = {}
    for name in lang_names:
        code = SUPPORTED_LANGUAGES.get(name)
        if code:
            results[name] = translate_text(text, code)
        else:
            results[name] = text
    return results


def get_language_names() -> list[str]:
    return list(SUPPORTED_LANGUAGES.keys())


def get_gtts_code(lang_name: str) -> str | None:
    """Return gTTS language code for a given display name, or None if not supported."""
    return GTTS_LANGUAGES.get(lang_name)



