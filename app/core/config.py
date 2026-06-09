"""
=============================================================================
 config.py — Configuraciones Globales del Sistema AVSR
=============================================================================
 Centraliza TODAS las constantes del sistema para evitar valores mágicos
 dispersos en el código. Usa dataclasses frozen (inmutables) para garantizar
 que ningún módulo modifique accidentalmente una configuración en runtime.

 Cada sección agrupa parámetros por dominio funcional:
   - Audio: formato del flujo PCM y tamaño de buffers.
   - Video: resolución esperada del frame de la cámara.
   - Whisper: hiperparámetros del modelo de transcripción.
   - MediaPipe: configuración del Face Mesh.
   - Fusión: umbrales de la estrategia Late Fusion.
   - Debug: flags para ventanas de visualización.
=============================================================================
"""

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Configuración de Audio
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AudioConfig:
    """Parámetros del flujo de audio PCM entrante."""

    # Frecuencia de muestreo en Hz.
    # 16 kHz es el estándar de Whisper; cualquier otro valor degrada calidad.
    SAMPLE_RATE: int = 16_000

    # Canales de audio. Mono = 1 (Whisper NO soporta estéreo).
    CHANNELS: int = 1

    # Bytes por muestra: PCM 16-bit = 2 bytes por sample.
    SAMPLE_WIDTH: int = 2

    # Duración en segundos de cada ventana de transcripción.
    # 3 segundos es un balance entre latencia (~3s) y contexto suficiente
    # para que Whisper produzca transcripciones coherentes.
    CHUNK_DURATION_S: float = 3.0

    # Tamaño máximo del buffer circular en segundos.
    # Evita que la memoria crezca indefinidamente si el procesamiento se atrasa.
    BUFFER_MAX_S: float = 30.0

    @property
    def chunk_bytes(self) -> int:
        """Bytes necesarios para llenar una ventana de transcripción."""
        return int(
            self.SAMPLE_RATE * self.CHANNELS * self.SAMPLE_WIDTH * self.CHUNK_DURATION_S
        )

    @property
    def buffer_max_bytes(self) -> int:
        """Límite máximo del buffer circular en bytes."""
        return int(
            self.SAMPLE_RATE * self.CHANNELS * self.SAMPLE_WIDTH * self.BUFFER_MAX_S
        )


# ─────────────────────────────────────────────────────────────────────────────
# Configuración de Video
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VideoConfig:
    """Parámetros del flujo de video entrante."""

    # Resolución esperada del frame (el cliente debe enviar a esta resolución).
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480


# ─────────────────────────────────────────────────────────────────────────────
# Configuración de Faster-Whisper
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class WhisperConfig:
    """
    Hiperparámetros para el modelo Faster-Whisper.

    IMPORTANTE — Restricciones de hardware AMD:
      - device="cpu": No hay soporte CUDA en GPUs AMD locales.
      - compute_type="int8": Cuantización de 8 bits para reducir uso de RAM
        y acelerar la inferencia en CPU (~2x más rápido que float32).
    """

    # Tamaño del modelo. "small" ofrece buena calidad en español con ~461M params.
    MODEL_SIZE: str = "small"

    # Dispositivo de inferencia. FORZADO a CPU para hardware AMD.
    DEVICE: str = "cpu"

    # Tipo de cómputo. int8 reduce el footprint de memoria a la mitad.
    COMPUTE_TYPE: str = "int8"

    # Idioma objetivo de la transcripción.
    LANGUAGE: str = "es"

    # Tamaño del beam search. 3 es un buen compromiso velocidad/calidad.
    # Valores más altos (5+) mejoran calidad pero aumentan latencia en CPU.
    BEAM_SIZE: int = 3

    # Activar el filtro VAD interno de Whisper para pre-segmentar silencios.
    VAD_FILTER: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Configuración de MediaPipe Face Mesh
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MediaPipeConfig:
    """
    Parámetros del detector de landmarks faciales.

    refine_landmarks=True activa el modelo de refinamiento de la zona
    peri-ocular y labial, aumentando la resolución de los landmarks de
    468 a 478 puntos con precisión submilimétrica.
    """

    # Número máximo de caras a detectar. 1 = solo el hablante principal.
    MAX_NUM_FACES: int = 1

    # Activar landmarks refinados (iris + contorno labial de alta precisión).
    REFINE_LANDMARKS: bool = True

    # Confianza mínima para la detección inicial de la cara.
    MIN_DETECTION_CONFIDENCE: float = 0.5

    # Confianza mínima para el tracking frame-a-frame (evita re-detección).
    MIN_TRACKING_CONFIDENCE: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Configuración de la Fusión Multimodal (Late Fusion)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FusionConfig:
    """
    Umbrales para la estrategia de fusión tardía Audio-Visual.

    La idea central:
      Si Whisper produce texto pero los labios estuvieron cerrados
      durante ese intervalo, el texto es probablemente una alucinación
      causada por ruido ambiental (soplos, estática, música de fondo).
    """

    # Umbral de Lip Opening Distance (LOD) en coordenadas normalizadas [0, 1].
    # Si LOD > LOD_THRESHOLD → labios abiertos → "hablando".
    # Si LOD <= LOD_THRESHOLD → labios cerrados → "silencio".
    #
    # 0.015 es conservador: detecta desde vocalizaciones sutiles.
    # Subir a 0.02-0.03 si hay muchos falsos positivos.
    LOD_THRESHOLD: float = 0.015

    # Ventana temporal de fusión en segundos.
    # Debe coincidir con la ventana de transcripción de audio.
    FUSION_WINDOW_S: float = 3.0

    # Ratio mínimo de frames con labios abiertos para aprobar la transcripción.
    # 0.3 = al menos el 30% de los frames deben mostrar actividad labial.
    MIN_SPEAKING_RATIO: float = 0.3

    # Umbral de confianza alta de Whisper para la regla de "penalización".
    # Si confianza > este valor Y el texto es sustancial, se penaliza
    # en vez de descartar (para no perder transcripciones legítimas).
    HIGH_CONFIDENCE_THRESHOLD: float = 0.9

    # Número mínimo de palabras para considerar un texto "sustancial".
    MIN_WORDS_SUBSTANTIAL: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# Configuración de Debug / Visualización
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DebugConfig:
    """Flags para herramientas de desarrollo y visualización."""

    # Factor de escala para ampliar el recorte de la región labial.
    # 3 = el crop de labios se mostrará 3x más grande que su tamaño real.
    LIP_CROP_SCALE: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Instancias globales (importables desde cualquier módulo)
# ─────────────────────────────────────────────────────────────────────────────
# Uso: from app.core.config import AUDIO_CFG, WHISPER_CFG, etc.

AUDIO_CFG = AudioConfig()
VIDEO_CFG = VideoConfig()
WHISPER_CFG = WhisperConfig()
MEDIAPIPE_CFG = MediaPipeConfig()
FUSION_CFG = FusionConfig()
DEBUG_CFG = DebugConfig()
