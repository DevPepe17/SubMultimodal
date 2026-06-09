"""Paquete models_ia — Singletons de modelos de Inteligencia Artificial."""

from app.models_ia.lip_reading_model import LipReadingModelSingleton
from app.models_ia.whisper_model import WhisperModelSingleton

__all__: list[str] = [
    "WhisperModelSingleton",
    "LipReadingModelSingleton",
]
