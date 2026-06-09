"""
=============================================================================
 audio_processor.py — Buffer de Audio + Inferencia de Faster-Whisper
=============================================================================
 Gestiona la acumulación de chunks de audio PCM en un buffer circular
 y lanza la transcripción con Whisper cuando se llena una ventana de
 CHUNK_DURATION_S segundos.

 Pipeline de procesamiento:
   1. Recibe bytes PCM crudos del WebSocket (16-bit, 16kHz, mono).
   2. Los acumula en un bytearray (buffer circular con límite).
   3. Al alcanzar chunk_bytes, extrae la ventana del buffer.
   4. Convierte a numpy float32 normalizado [-1.0, 1.0].
   5. Lanza la transcripción en un ThreadPoolExecutor para no bloquear
      el event loop de asyncio.
   6. Devuelve un AudioResult con texto, confianza y timestamps.

 ¿Por qué un ThreadPoolExecutor?
   - Whisper es CPU-bound (cálculo intensivo de matrices).
   - asyncio es para I/O-bound (esperar red, disco, etc.).
   - Si ejecutamos Whisper directamente en el event loop, BLOQUEAMOS
     todas las corrutinas concurrentes (incluyendo la recepción de
     frames de video del WebSocket).
   - run_in_executor delega a un hilo del OS, liberando el event loop.
=============================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.core.config import AUDIO_CFG
from app.models_ia.whisper_model import WhisperModelSingleton

# Logger del módulo
logger = logging.getLogger(__name__)

# Pool de hilos compartido para inferencia de Whisper.
# max_workers=1 porque Whisper ya usa múltiples threads internos (CTranslate2).
# Más de 1 worker causaría contención de CPU sin beneficio.
_whisper_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")


# ─────────────────────────────────────────────────────────────────────────────
# Estructura de resultado de audio
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AudioResult:
    """
    Resultado de una transcripción de audio.

    Attributes:
        texto: Texto transcrito concatenado de todos los segmentos.
        confianza: Promedio de log-probabilidades de los segmentos.
                   Valores típicos: -0.1 (alta) a -1.0 (baja).
        timestamp_inicio: Momento (time.time()) en que se empezó a acumular
                          el buffer que generó esta transcripción.
        timestamp_fin: Momento en que se completó la acumulación del buffer.
        segmentos: Lista cruda de segmentos devueltos por Whisper.
    """

    texto: str = ""
    confianza: float = 0.0
    timestamp_inicio: float = 0.0
    timestamp_fin: float = 0.0
    segmentos: list[dict] = field(default_factory=list)

    @property
    def tiene_texto(self) -> bool:
        """True si la transcripción produjo texto no vacío."""
        return len(self.texto.strip()) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Procesador de Audio
# ─────────────────────────────────────────────────────────────────────────────
class AudioProcessor:
    """
    Acumula audio PCM en un buffer circular y transcribe con Whisper
    cuando se llena una ventana de CHUNK_DURATION_S segundos.
    """

    def __init__(self) -> None:
        """Inicializa el buffer de audio y la referencia al modelo Whisper."""

        # Buffer circular: bytearray mutable para acumulación eficiente.
        self._buffer: bytearray = bytearray()

        # Timestamp del primer byte del chunk actual (para sincronización con video).
        self._chunk_start_time: float = time.time()

        # Referencia al Singleton de Whisper (ya cargado en memoria).
        self._whisper: WhisperModelSingleton = WhisperModelSingleton()

        # Flag para saber si hay una transcripción en progreso.
        self._transcribing: bool = False

        logger.debug(
            "AudioProcessor inicializado | chunk=%d bytes (%.1fs) | max_buffer=%d bytes",
            AUDIO_CFG.chunk_bytes,
            AUDIO_CFG.CHUNK_DURATION_S,
            AUDIO_CFG.buffer_max_bytes,
        )

    def add_audio_chunk(self, pcm_bytes: bytes) -> None:
        """
        Agrega un chunk de bytes PCM al buffer circular.

        Si el buffer excede BUFFER_MAX_S, descarta los bytes más antiguos
        para evitar crecimiento indefinido de memoria.

        Args:
            pcm_bytes: Bytes de audio PCM 16-bit, little-endian, mono, 16kHz.
        """
        self._buffer.extend(pcm_bytes)

        # Aplicar límite circular: descartar bytes antiguos si excede el máximo
        if len(self._buffer) > AUDIO_CFG.buffer_max_bytes:
            exceso = len(self._buffer) - AUDIO_CFG.buffer_max_bytes
            self._buffer = self._buffer[exceso:]
            logger.warning(
                "Buffer de audio excedió el límite. Descartados %d bytes antiguos.",
                exceso,
            )

    async def process_if_ready(self) -> Optional[AudioResult]:
        """
        Verifica si hay suficiente audio para transcribir y lanza la inferencia.

        Returns:
            AudioResult con la transcripción si había suficiente audio,
            None si el buffer aún no alcanza CHUNK_DURATION_S o si ya hay
            una transcripción en progreso.
        """
        # No lanzar transcripción duplicada si ya hay una en curso
        if self._transcribing:
            return None

        # Verificar si el buffer tiene suficiente audio
        if len(self._buffer) < AUDIO_CFG.chunk_bytes:
            return None

        # ─── Extraer la ventana de audio del buffer ───
        audio_bytes: bytes = bytes(self._buffer[: AUDIO_CFG.chunk_bytes])
        self._buffer = self._buffer[AUDIO_CFG.chunk_bytes :]

        timestamp_fin = time.time()
        timestamp_inicio = self._chunk_start_time
        self._chunk_start_time = timestamp_fin  # Reset para el siguiente chunk

        # ─── Convertir PCM 16-bit a float32 normalizado ───
        # PCM 16-bit: cada sample es un int16 en rango [-32768, 32767].
        # Whisper espera float32 en rango [-1.0, 1.0].
        # Dividir por 32768.0 normaliza al rango correcto.
        audio_array: np.ndarray = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # ─── Lanzar inferencia en ThreadPoolExecutor ───
        self._transcribing = True
        try:
            loop = asyncio.get_running_loop()
            segmentos: list[dict] = await loop.run_in_executor(
                _whisper_executor,
                self._whisper.transcribe,
                audio_array,
            )
        except Exception as e:
            logger.error("Error en la inferencia de Whisper: %s", str(e), exc_info=True)
            self._transcribing = False
            return None
        finally:
            self._transcribing = False

        # ─── Construir el resultado ───
        if not segmentos:
            return AudioResult(
                texto="",
                confianza=0.0,
                timestamp_inicio=timestamp_inicio,
                timestamp_fin=timestamp_fin,
                segmentos=[],
            )

        # Concatenar textos de todos los segmentos
        texto_completo = " ".join(s["texto"] for s in segmentos if s["texto"])

        # Calcular confianza promedio (log-prob → convertir a probabilidad lineal)
        # avg_log_prob de Whisper está en log natural. exp() lo convierte a [0, 1].
        confianzas = [s["confianza"] for s in segmentos]
        confianza_promedio = float(np.mean(confianzas)) if confianzas else 0.0

        return AudioResult(
            texto=texto_completo,
            confianza=confianza_promedio,
            timestamp_inicio=timestamp_inicio,
            timestamp_fin=timestamp_fin,
            segmentos=segmentos,
        )

    def reset(self) -> None:
        """Limpia el buffer de audio y resetea el estado."""
        self._buffer.clear()
        self._chunk_start_time = time.time()
        self._transcribing = False
        logger.debug("AudioProcessor reseteado.")

    @property
    def buffer_duration_s(self) -> float:
        """Duración actual del audio en el buffer, en segundos."""
        return len(self._buffer) / (
            AUDIO_CFG.SAMPLE_RATE * AUDIO_CFG.CHANNELS * AUDIO_CFG.SAMPLE_WIDTH
        )
