import pyttsx3

_engine = None


def _get_engine() -> pyttsx3.Engine:
    """Lazily initialise and cache the TTS engine."""
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 175)    # words per minute
        _engine.setProperty("volume", 1.0)  # 0.0 – 1.0
    return _engine


def speak_streaming(text: str) -> None:
    """Speak text aloud using the system TTS engine.
    Named 'speak_streaming' to match the import in rag_pipeline.py.
    """
    if not text or not text.strip():
        return
    engine = _get_engine()
    engine.say(text)
    engine.runAndWait()