"""Paquete core — configuraciones globales del sistema AVSR."""

from app.core.config import (
    AUDIO_CFG,
    DEBUG_CFG,
    FUSION_CFG,
    MEDIAPIPE_CFG,
    VIDEO_CFG,
    WHISPER_CFG,
)

__all__: list[str] = [
    "AUDIO_CFG",
    "VIDEO_CFG",
    "WHISPER_CFG",
    "MEDIAPIPE_CFG",
    "FUSION_CFG",
    "DEBUG_CFG",
]
