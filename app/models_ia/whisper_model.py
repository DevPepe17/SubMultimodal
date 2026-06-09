"""
=============================================================================
 whisper_model.py — Singleton Thread-Safe para Faster-Whisper
=============================================================================
 Carga el modelo Faster-Whisper UNA SOLA VEZ en memoria RAM y reutiliza
 la misma instancia para todas las conexiones WebSocket concurrentes.

 ¿Por qué Singleton?
   - El modelo "small" ocupa ~500 MB en RAM (con int8).
   - Recargar el modelo en cada conexión causaría picos de latencia de
     5-10 segundos y eventualmente OOM (Out Of Memory) en CPUs AMD modestas.
   - Con Singleton, la primera conexión paga el costo de carga (~5s) y
     todas las siguientes obtienen el modelo de forma instantánea.

 Thread-Safety:
   - Se usa threading.Lock para evitar race conditions si dos conexiones
     WebSocket intentan instanciar el Singleton simultáneamente.
   - La inferencia en sí (transcribe) es thread-safe en Faster-Whisper
     porque CTranslate2 maneja su propio pool de threads interno.
=============================================================================
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from app.core.config import WHISPER_CFG

# Logger del módulo
logger = logging.getLogger(__name__)


class WhisperModelSingleton:
    """
    Wrapper Singleton para el modelo Faster-Whisper.

    Atributos de clase:
        _instance: Referencia única a la instancia del Singleton.
        _lock: Mutex para garantizar thread-safety en la creación.
        _initialized: Flag para evitar re-inicialización del __init__.
    """

    _instance: Optional[WhisperModelSingleton] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> WhisperModelSingleton:
        """
        Controla la creación de la instancia única.

        Usa double-checked locking:
          1. Primer check sin lock (fast path para el 99% de las llamadas).
          2. Si _instance es None, adquiere el lock y verifica de nuevo.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check: otro hilo pudo crear la instancia mientras esperábamos el lock
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        Inicializa el modelo Faster-Whisper con los parámetros de config.

        Se ejecuta solo una vez gracias al flag _initialized.
        Esto es necesario porque __init__ se llama SIEMPRE después de __new__,
        incluso cuando __new__ devuelve una instancia ya existente.
        """
        if WhisperModelSingleton._initialized:
            return

        logger.info(
            "═══ Cargando modelo Faster-Whisper ═══ "
            "modelo=%s | dispositivo=%s | precisión=%s",
            WHISPER_CFG.MODEL_SIZE,
            WHISPER_CFG.DEVICE,
            WHISPER_CFG.COMPUTE_TYPE,
        )

        try:
            # ─── Inicialización del modelo CTranslate2 ───
            # device="cpu" → forzado para AMD (sin CUDA).
            # compute_type="int8" → cuantización de 8 bits:
            #   - Reduce uso de RAM ~50% vs float32.
            #   - Acelera inferencia ~2x en CPU (instrucciones VNNI/SSE).
            #   - Degradación de calidad negligible para modelo "small".
            self._model: WhisperModel = WhisperModel(
                model_size_or_path=WHISPER_CFG.MODEL_SIZE,
                device=WHISPER_CFG.DEVICE,
                compute_type=WHISPER_CFG.COMPUTE_TYPE,
            )

            WhisperModelSingleton._initialized = True
            logger.info("✓ Modelo Faster-Whisper cargado exitosamente en RAM.")

        except Exception as e:
            # Si la carga falla, reseteamos el Singleton para permitir reintento
            WhisperModelSingleton._instance = None
            logger.critical(
                "✗ Error CRÍTICO al cargar Faster-Whisper: %s", str(e), exc_info=True
            )
            raise RuntimeError(
                f"No se pudo cargar el modelo Whisper '{WHISPER_CFG.MODEL_SIZE}': {e}"
            ) from e

    def transcribe(self, audio_array: np.ndarray) -> list[dict]:
        """
        Transcribe un array de audio usando Faster-Whisper.

        Args:
            audio_array: Array NumPy de float32 normalizado en rango [-1.0, 1.0],
                         muestreado a 16 kHz, mono.

        Returns:
            Lista de diccionarios con las transcripciones:
            [
                {
                    "texto": "hola mundo",
                    "inicio": 0.0,    # segundos desde el inicio del chunk
                    "fin": 1.5,
                    "confianza": 0.92  # probabilidad promedio del segmento
                },
                ...
            ]

        Raises:
            RuntimeError: Si el modelo no está inicializado correctamente.
        """
        if not WhisperModelSingleton._initialized:
            raise RuntimeError("El modelo Whisper no ha sido inicializado.")

        resultados: list[dict] = []

        try:
            # ─── Inferencia de Faster-Whisper ───
            # segments es un generador lazy; info contiene metadatos globales.
            # language="es" → fuerza español (evita auto-detección costosa).
            # beam_size=3 → compromiso velocidad/calidad razonable para CPU.
            # vad_filter=True → pre-filtra silencios antes de transcribir,
            #   reduciendo alucinaciones en segmentos sin habla.
            segments, info = self._model.transcribe(
                audio_array,
                language=WHISPER_CFG.LANGUAGE,
                beam_size=WHISPER_CFG.BEAM_SIZE,
                vad_filter=WHISPER_CFG.VAD_FILTER,
            )

            # Materializar el generador de segmentos
            for segmento in segments:
                resultados.append(
                    {
                        "texto": segmento.text.strip(),
                        "inicio": segmento.start,
                        "fin": segmento.end,
                        "confianza": segmento.avg_logprob,
                    }
                )

            logger.debug(
                "Whisper transcribió %d segmento(s) | idioma_detectado=%s | prob=%.2f",
                len(resultados),
                info.language,
                info.language_probability,
            )

        except Exception as e:
            logger.error(
                "Error durante la transcripción de Whisper: %s", str(e), exc_info=True
            )

        return resultados

    @property
    def is_loaded(self) -> bool:
        """Verifica si el modelo está cargado en memoria."""
        return WhisperModelSingleton._initialized
