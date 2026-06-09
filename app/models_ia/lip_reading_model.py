"""
=============================================================================
 lip_reading_model.py — Singleton Thread-Safe para MediaPipe FaceLandmarker
=============================================================================
 Carga el detector de landmarks faciales UNA SOLA VEZ y lo reutiliza
 para todas las sesiones de WebSocket.

 NOTA SOBRE MEDIAPIPE 0.10.35+:
   Las versiones recientes de MediaPipe eliminaron la API legacy
   `mp.solutions.face_mesh`. Ahora se usa la nueva Tasks API:
     - mediapipe.tasks.python.vision.FaceLandmarker
     - Requiere un archivo de modelo descargado: face_landmarker.task
     - RunningMode.VIDEO para procesamiento frame-a-frame con tracking.

 FaceLandmarker con el modelo float16:
   - Detecta 478 landmarks faciales 3D (equivalente a refine_landmarks=True
     de la API legacy, que incluye iris y contorno labial de alta precisión).
   - Esto es CRÍTICO para calcular el LOD (Lip Opening Distance) con
     la resolución necesaria para distinguir habla de silencio.

 NOTA sobre thread-safety de MediaPipe:
   - FaceLandmarker NO es thread-safe internamente. Cada llamada a
     detect_for_video() debe ser serializada. Usamos un Lock dedicado.
=============================================================================
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

import numpy as np

from app.core.config import MEDIAPIPE_CFG

# Logger del módulo
logger = logging.getLogger(__name__)

# Ruta al archivo del modelo descargado.
# Se busca primero en models_weight/ (directorio del proyecto).
_MODEL_FILENAME = "face_landmarker.task"
_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models_weight",
)
_MODEL_PATH = os.path.join(_MODEL_DIR, _MODEL_FILENAME)


class LipReadingModelSingleton:
    """
    Wrapper Singleton para MediaPipe FaceLandmarker (Tasks API).

    Encapsula la carga del modelo y la inferencia de landmarks faciales,
    exponiendo una interfaz limpia para el VideoProcessor.
    """

    _instance: Optional[LipReadingModelSingleton] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> LipReadingModelSingleton:
        """Creación thread-safe con double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        Inicializa MediaPipe FaceLandmarker con la Tasks API.

        Usa RunningMode.VIDEO para procesamiento frame-a-frame con tracking
        temporal (más eficiente que IMAGE porque reutiliza la detección
        anterior para predecir la posición de la cara en el frame siguiente).
        """
        if LipReadingModelSingleton._initialized:
            return

        logger.info(
            "═══ Cargando MediaPipe FaceLandmarker (Tasks API) ═══ "
            "max_faces=%d | det_conf=%.1f | track_conf=%.1f",
            MEDIAPIPE_CFG.MAX_NUM_FACES,
            MEDIAPIPE_CFG.MIN_DETECTION_CONFIDENCE,
            MEDIAPIPE_CFG.MIN_TRACKING_CONFIDENCE,
        )

        # Verificar que el archivo del modelo existe
        if not os.path.isfile(_MODEL_PATH):
            error_msg = (
                f"No se encontró el modelo FaceLandmarker en: {_MODEL_PATH}\n"
                "Descárgalo con:\n"
                "  curl -o models_weight/face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
            logger.critical(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
            )
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )

            # ─── Configuración de opciones ───
            # RunningMode.VIDEO: procesa frames secuenciales con tracking temporal.
            #   - Requiere timestamps monotónicamente crecientes.
            #   - Más eficiente que IMAGE porque usa tracking en vez de re-detección.
            # num_faces=1: solo el hablante principal.
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_faces=MEDIAPIPE_CFG.MAX_NUM_FACES,
                min_face_detection_confidence=MEDIAPIPE_CFG.MIN_DETECTION_CONFIDENCE,
                min_face_presence_confidence=MEDIAPIPE_CFG.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MEDIAPIPE_CFG.MIN_TRACKING_CONFIDENCE,
                output_face_blendshapes=False,  # No necesitamos blendshapes
                output_facial_transformation_matrixes=False,
            )

            # ─── Crear el FaceLandmarker ───
            self._landmarker: FaceLandmarker = FaceLandmarker.create_from_options(
                options
            )

            # Guardar referencia a la clase Image de MediaPipe para conversiones
            self._mp_image_cls = mp.Image
            self._mp_image_format = mp.ImageFormat

            # Lock dedicado para serializar llamadas a detect_for_video()
            self._inference_lock: threading.Lock = threading.Lock()

            # Contador de timestamps para el modo VIDEO (debe ser monótonamente creciente)
            self._frame_timestamp_ms: int = 0

            LipReadingModelSingleton._initialized = True
            logger.info("✓ MediaPipe FaceLandmarker cargado exitosamente.")

        except Exception as e:
            LipReadingModelSingleton._instance = None
            logger.critical(
                "✗ Error CRÍTICO al cargar MediaPipe: %s", str(e), exc_info=True
            )
            raise RuntimeError(
                f"No se pudo cargar MediaPipe FaceLandmarker: {e}"
            ) from e

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[Any]:
        """
        Procesa un frame RGB y devuelve los landmarks faciales detectados.

        Args:
            frame_rgb: Frame de video en formato RGB (NO BGR), shape (H, W, 3),
                       dtype uint8, valores en rango [0, 255].

        Returns:
            El objeto face_landmarks[0] (lista de NormalizedLandmark) si se
            detectó una cara, o None si no se detectó ninguna cara.

            Cada NormalizedLandmark tiene:
              - .x → posición horizontal normalizada [0, 1]
              - .y → posición vertical normalizada [0, 1]
              - .z → profundidad relativa
        """
        if not LipReadingModelSingleton._initialized:
            raise RuntimeError("MediaPipe FaceLandmarker no ha sido inicializado.")

        try:
            with self._inference_lock:
                # Convertir numpy array a mediapipe.Image
                mp_image = self._mp_image_cls(
                    image_format=self._mp_image_format.SRGB,
                    data=frame_rgb,
                )

                # Incrementar timestamp (debe ser monótonamente creciente)
                self._frame_timestamp_ms += 33  # ~30 FPS (1000ms / 30 ≈ 33ms)

                # Detectar landmarks en modo VIDEO (con tracking temporal)
                result = self._landmarker.detect_for_video(
                    mp_image, self._frame_timestamp_ms
                )

            # Verificar si se detectó al menos una cara
            if result.face_landmarks and len(result.face_landmarks) > 0:
                return result.face_landmarks[0]  # Lista de NormalizedLandmark

            return None

        except Exception as e:
            logger.error(
                "Error al procesar frame con MediaPipe: %s", str(e), exc_info=True
            )
            return None

    @property
    def is_loaded(self) -> bool:
        """Verifica si el modelo está cargado en memoria."""
        return LipReadingModelSingleton._initialized

    def close(self) -> None:
        """Libera los recursos de MediaPipe de forma limpia."""
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            try:
                self._landmarker.close()
                logger.info("MediaPipe FaceLandmarker cerrado correctamente.")
            except Exception as e:
                logger.warning("Error al cerrar MediaPipe: %s", str(e))
