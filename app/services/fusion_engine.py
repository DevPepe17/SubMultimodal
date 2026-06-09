"""
=============================================================================
 fusion_engine.py — Motor de Fusión Tardía (Late Fusion) Audio-Visual
=============================================================================
 Este módulo es el "cerebro decisional" del sistema AVSR. Recibe las
 transcripciones de Whisper y las correlaciona temporalmente con el
 estado de actividad labial detectado por el VideoProcessor.

 ═══════════════════════════════════════════════════════════════════════
 ESTRATEGIA DE FUSIÓN TARDÍA (LATE FUSION)
 ═══════════════════════════════════════════════════════════════════════

 ¿Qué es Late Fusion?
   En sistemas multimodales, hay tres estrategias de fusión:
     - Early Fusion: combinar features ANTES de la inferencia.
     - Mid Fusion: combinar features en capas intermedias del modelo.
     - Late Fusion: combinar RESULTADOS de inferencias independientes.

   Nosotros usamos Late Fusion porque:
     1. No modificamos los modelos internos (Whisper ni MediaPipe).
     2. Cada modalidad se procesa con su propio modelo especializado.
     3. Solo en la etapa final decidimos si la transcripción es válida.

 ¿Por qué funciona contra las alucinaciones?
   Whisper es propenso a "alucinar" texto cuando recibe ruido que
   se parece vagamente al habla (soplos, estática, música).
   Pero si la cámara muestra que los labios estaban CERRADOS durante
   ese intervalo, es casi seguro que el texto es una alucinación.

 Reglas de decisión:
   1. APROBADA: speaking_ratio ≥ MIN_SPEAKING_RATIO (30%)
      → Los labios se movieron suficiente → texto probablemente real.

   2. PENALIZADA: speaking_ratio < 30% PERO confianza de Whisper > 0.9
      Y texto tiene > 5 palabras.
      → Caso borde: Whisper está muy seguro pero los labios no se movieron.
      → Marcamos con fusion_aprobada=False pero NO descartamos el texto.
      → El consumidor decide (podría ser habla con la boca tapada, etc.).

   3. DESCARTADA: speaking_ratio < 30% Y no cumple regla 2.
      → Casi seguro es una alucinación → texto descartado.

 ═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from app.core.config import FUSION_CFG
from app.services.audio_processor import AudioResult
from app.services.video_processor import VideoProcessor

# Logger del módulo
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Estructura de resultado de fusión
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FusionResult:
    """
    Resultado final del proceso de fusión audio-visual.

    Este es el objeto que se serializa a JSON y se envía al cliente
    a través del WebSocket.

    Attributes:
        texto: Texto transcrito (vacío si fue descartado).
        confianza_audio: Log-probabilidad promedio de Whisper.
        estado_labios: "hablando" o "silencio" (estado dominante).
        lod_promedio: LOD promedio en la ventana de fusión.
        speaking_ratio: Ratio de frames con labios abiertos [0, 1].
        fusion_aprobada: True si la transcripción pasó el filtro de fusión.
        motivo: Explicación de la decisión ("aprobada", "penalizada", "descartada").
        timestamp: Momento de la decisión de fusión.
    """

    texto: str = ""
    confianza_audio: float = 0.0
    estado_labios: str = "silencio"
    lod_promedio: float = 0.0
    speaking_ratio: float = 0.0
    fusion_aprobada: bool = False
    motivo: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convierte a diccionario serializable para JSON."""
        return {
            "tipo": "transcripcion",
            "texto": self.texto,
            "confianza_audio": round(self.confianza_audio, 4),
            "estado_labios": self.estado_labios,
            "lod_promedio": round(self.lod_promedio, 6),
            "speaking_ratio": round(self.speaking_ratio, 3),
            "fusion_aprobada": self.fusion_aprobada,
            "motivo": self.motivo,
            "timestamp": round(self.timestamp, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Motor de Fusión
# ─────────────────────────────────────────────────────────────────────────────
class FusionEngine:
    """
    Motor de fusión tardía que correlaciona transcripciones de audio
    con actividad labial para filtrar alucinaciones.
    """

    def __init__(self, video_processor: VideoProcessor) -> None:
        """
        Inicializa el motor de fusión.

        Args:
            video_processor: Referencia al VideoProcessor de la sesión actual.
                             Se usa para consultar el buffer de estados labiales.
        """
        self._video_processor: VideoProcessor = video_processor

        # Contador de estadísticas de la sesión
        self._stats = {
            "aprobadas": 0,
            "penalizadas": 0,
            "descartadas": 0,
            "vacias": 0,
        }

        logger.debug(
            "FusionEngine inicializado | umbral_LOD=%.4f | min_speaking_ratio=%.2f",
            FUSION_CFG.LOD_THRESHOLD,
            FUSION_CFG.MIN_SPEAKING_RATIO,
        )

    def evaluate(self, audio_result: AudioResult) -> FusionResult:
        """
        Evalúa una transcripción de audio contra la actividad labial.

        Este es el método principal del motor de fusión. Implementa las
        3 reglas de decisión documentadas en el encabezado del módulo.

        Args:
            audio_result: Resultado de la transcripción de Whisper, incluyendo
                          texto, confianza y ventana temporal [t_inicio, t_fin].

        Returns:
            FusionResult con la decisión final (aprobada/penalizada/descartada).
        """
        ahora = time.time()

        # ─── Caso trivial: Whisper no produjo texto ───
        if not audio_result.tiene_texto:
            self._stats["vacias"] += 1
            logger.debug("Fusión: transcripción vacía → ignorada.")
            return FusionResult(
                texto="",
                confianza_audio=audio_result.confianza,
                estado_labios="silencio",
                lod_promedio=0.0,
                speaking_ratio=0.0,
                fusion_aprobada=False,
                motivo="transcripcion_vacia",
                timestamp=ahora,
            )

        # ─── Consultar el buffer de estados labiales ───
        # Usamos la ventana temporal del audio para buscar estados de labios
        # que correspondan al mismo intervalo de tiempo.
        t_start = audio_result.timestamp_inicio
        t_end = audio_result.timestamp_fin

        speaking_ratio = self._video_processor.get_speaking_ratio(t_start, t_end)
        lod_promedio = self._video_processor.get_average_lod(t_start, t_end)

        # Estado dominante basado en el ratio
        estado_labios = (
            "hablando"
            if speaking_ratio >= FUSION_CFG.MIN_SPEAKING_RATIO
            else "silencio"
        )

        # Convertir log-probabilidad a probabilidad lineal para la regla de confianza.
        # avg_log_prob de Whisper está en log natural (negativo).
        # exp(avg_log_prob) ∈ (0, 1] → probabilidad lineal.
        confianza_lineal = (
            math.exp(audio_result.confianza)
            if audio_result.confianza < 0
            else audio_result.confianza
        )

        # Contar palabras en la transcripción
        num_palabras = len(audio_result.texto.split())

        # ─── Regla 1: APROBADA ───
        # Los labios se movieron suficiente durante la ventana de audio.
        if speaking_ratio >= FUSION_CFG.MIN_SPEAKING_RATIO:
            self._stats["aprobadas"] += 1
            logger.info(
                "✅ APROBADA | texto='%s' | ratio=%.2f | LOD=%.4f | conf=%.3f",
                audio_result.texto[:50],
                speaking_ratio,
                lod_promedio,
                confianza_lineal,
            )
            return FusionResult(
                texto=audio_result.texto,
                confianza_audio=confianza_lineal,
                estado_labios=estado_labios,
                lod_promedio=lod_promedio,
                speaking_ratio=speaking_ratio,
                fusion_aprobada=True,
                motivo="aprobada",
                timestamp=ahora,
            )

        # ─── Regla 2: PENALIZADA ───
        # Los labios NO se movieron, pero Whisper tiene alta confianza
        # y el texto es sustancial. No descartamos, pero marcamos.
        if (
            confianza_lineal > FUSION_CFG.HIGH_CONFIDENCE_THRESHOLD
            and num_palabras >= FUSION_CFG.MIN_WORDS_SUBSTANTIAL
        ):
            self._stats["penalizadas"] += 1
            logger.warning(
                "⚠️ PENALIZADA | texto='%s' | ratio=%.2f | conf=%.3f | palabras=%d",
                audio_result.texto[:50],
                speaking_ratio,
                confianza_lineal,
                num_palabras,
            )
            return FusionResult(
                texto=audio_result.texto,
                confianza_audio=confianza_lineal,
                estado_labios=estado_labios,
                lod_promedio=lod_promedio,
                speaking_ratio=speaking_ratio,
                fusion_aprobada=False,
                motivo="penalizada_alta_confianza_labios_cerrados",
                timestamp=ahora,
            )

        # ─── Regla 3: DESCARTADA ───
        # Los labios cerrados + baja confianza o texto corto = alucinación.
        self._stats["descartadas"] += 1
        logger.info(
            "❌ DESCARTADA | texto='%s' | ratio=%.2f | conf=%.3f | palabras=%d",
            audio_result.texto[:50],
            speaking_ratio,
            confianza_lineal,
            num_palabras,
        )
        return FusionResult(
            texto="",
            confianza_audio=confianza_lineal,
            estado_labios=estado_labios,
            lod_promedio=lod_promedio,
            speaking_ratio=speaking_ratio,
            fusion_aprobada=False,
            motivo="descartada_labios_cerrados",
            timestamp=ahora,
        )

    @property
    def stats(self) -> dict:
        """Devuelve las estadísticas acumuladas de la sesión de fusión."""
        return dict(self._stats)

    def reset(self) -> None:
        """Resetea las estadísticas de la sesión."""
        for key in self._stats:
            self._stats[key] = 0
        logger.debug("FusionEngine: estadísticas reseteadas.")
