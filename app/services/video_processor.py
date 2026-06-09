"""
=============================================================================
 video_processor.py — Procesamiento de Frames + MediaPipe FaceLandmarker + LOD
=============================================================================
 Este módulo es el "ojo" del sistema AVSR. Procesa cada frame de video
 para extraer la actividad labial del hablante y determinar si está
 hablando o en silencio.

 Pipeline de procesamiento por frame:
   1. Decodifica bytes JPEG → numpy array BGR (OpenCV).
   2. Convierte BGR → RGB (MediaPipe requiere RGB).
   3. Detecta landmarks faciales con FaceLandmarker (478 puntos 3D).
   4. Extrae los landmarks del contorno labial (interno y externo).
   5. Calcula el LOD (Lip Opening Distance) usando 3 pares verticales.
   6. Compara LOD con el umbral para determinar estado: hablando/silencio.
   7. Almacena el estado en un buffer temporal para consulta del FusionEngine.
   8. (Opcional) Muestra ventana de debug con landmarks y crop de labios.

 ═══════════════════════════════════════════════════════════════════════
 MATEMÁTICAS DEL LOD (Lip Opening Distance)
 ═══════════════════════════════════════════════════════════════════════

 MediaPipe FaceLandmarker devuelve 478 landmarks normalizados en [0, 1]:
   - landmark.x ∈ [0, 1] → posición horizontal
   - landmark.y ∈ [0, 1] → posición vertical
   - landmark.z → profundidad relativa (no la usamos para LOD 2D)

 Landmarks del contorno labial INTERNO:
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Labio superior interno:                                           │
 │    78 → 191 → 80 → 81 → 82 → 13 → 312 → 311 → 310 → 415 → 308  │
 │                                                                    │
 │  Labio inferior interno:                                           │
 │    78 → 95 → 88 → 178 → 87 → 14 → 317 → 402 → 318 → 324 → 308  │
 └────────────────────────────────────────────────────────────────────┘

 Para calcular la apertura vertical, usamos 3 PARES de landmarks
 que conectan el labio superior interno con el inferior interno:

   Par 1 (centro):           L13  ↔ L14
   Par 2 (izquierda):        L81  ↔ L178
   Par 3 (derecha):          L311 ↔ L402

 Cada par mide la distancia euclidiana 2D:

   d(a, b) = √[(ax - bx)² + (ay - by)²]

 El LOD es el PROMEDIO de las 3 distancias:

   LOD = (1/3) × [d(L13, L14) + d(L81, L178) + d(L311, L402)]

 ¿Por qué 3 pares y no solo 1?
   - La boca no se abre simétricamente al hablar.
   - Al decir "O", el centro se abre más que los lados.
   - Al decir "E", los lados se estiran más que el centro.
   - Promediar 3 puntos captura estas variaciones fonéticas.

 ¿Por qué coordenadas normalizadas?
   - El LOD es INVARIANTE al tamaño del frame (640x480 vs 1920x1080
     darán el mismo LOD porque las coords están en [0, 1]).
   - Esto hace que el umbral LOD_THRESHOLD sea universal.
 ═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.core.config import DEBUG_CFG, FUSION_CFG
from app.models_ia.lip_reading_model import LipReadingModelSingleton

# Logger del módulo
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constantes de landmarks labiales de MediaPipe
# ─────────────────────────────────────────────────────────────────────────────

# Índices de los landmarks del contorno labial EXTERNO (para dibujo y crop).
LABIO_EXTERNO_INDICES: list[int] = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
]

# Índices del contorno labial INTERNO (para cálculo de LOD).
LABIO_SUPERIOR_INTERNO: list[int] = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LABIO_INFERIOR_INTERNO: list[int] = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Pares verticales para el cálculo del LOD.
# Cada tupla (superior, inferior) representa un par de landmarks que se
# enfrentan verticalmente a través de la apertura de la boca.
PARES_LOD: list[tuple[int, int]] = [
    (13, 14),  # Centro de la boca
    (81, 178),  # Izquierda del centro
    (311, 402),  # Derecha del centro
]

# Todos los índices de labios (para dibujar y calcular bounding box del crop).
TODOS_LABIOS_INDICES: list[int] = list(
    set(LABIO_EXTERNO_INDICES + LABIO_SUPERIOR_INTERNO + LABIO_INFERIOR_INTERNO)
)


# ─────────────────────────────────────────────────────────────────────────────
# Estructura de estado labial
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LipState:
    """
    Estado de los labios en un instante de tiempo.

    Attributes:
        timestamp: Momento (time.time()) en que se procesó el frame.
        lod: Lip Opening Distance calculada (coordenadas normalizadas).
        is_speaking: True si LOD > LOD_THRESHOLD (labios abiertos).
        landmarks_pixels: Coordenadas (x, y) de los landmarks labiales
                          en píxeles (para dibujo en debug).
    """

    timestamp: float
    lod: float
    is_speaking: bool
    landmarks_pixels: Optional[list[tuple[int, int]]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Procesador de Video
# ─────────────────────────────────────────────────────────────────────────────
class VideoProcessor:
    """
    Procesa frames de video para detectar actividad labial.

    Mantiene un buffer temporal de estados labiales (deque) consultable
    por el FusionEngine para correlacionar con las transcripciones de audio.
    """

    def __init__(self) -> None:
        """Inicializa el procesador de video y el buffer de estados."""

        # Singleton de MediaPipe FaceLandmarker (ya cargado en memoria).
        self._face_mesh: LipReadingModelSingleton = LipReadingModelSingleton()

        # Buffer temporal de estados labiales.
        # maxlen = estimación de frames en la ventana de fusión.
        # A ~30 FPS y 3s de ventana → ~90 frames. Usamos 300 para margen.
        self._lip_states: deque[LipState] = deque(maxlen=300)

        # Último frame procesado (para debug visual).
        self._last_frame_bgr: Optional[np.ndarray] = None
        self._last_lip_crop: Optional[np.ndarray] = None
        self.latest_debug_jpeg: Optional[bytes] = None

        logger.debug("VideoProcessor inicializado con buffer de %d estados.", 300)

    def process_frame_bytes(self, jpeg_bytes: bytes) -> Optional[LipState]:
        """
        Procesa bytes JPEG de un frame de video.

        Args:
            jpeg_bytes: Frame codificado en formato JPEG.

        Returns:
            LipState con el estado actual de los labios, o None si
            no se pudo decodificar el frame o detectar una cara.
        """
        # ─── Paso 1: Decodificar JPEG → numpy BGR ───
        frame_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            logger.warning("No se pudo decodificar el frame JPEG recibido.")
            return None

        # ─── Paso 2: Convertir BGR → RGB (MediaPipe requiere RGB) ───
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ─── Paso 3: Detectar landmarks con FaceLandmarker ───
        # La nueva Tasks API devuelve una lista de NormalizedLandmark.
        face_landmarks = self._face_mesh.process_frame(frame_rgb)

        if face_landmarks is None:
            # No se detectó cara → asumir silencio
            state = LipState(
                timestamp=time.time(),
                lod=0.0,
                is_speaking=False,
            )
            self._lip_states.append(state)
            self._last_frame_bgr = frame_bgr
            self._last_lip_crop = None

            # Generar frame de debug sin landmarks
            self._generate_debug_frame(frame_bgr, state, landmarks=None)

            return state

        # ─── Paso 4: Calcular LOD (Lip Opening Distance) ───
        lod = self._calculate_lod(face_landmarks)

        # ─── Paso 5: Determinar estado de habla ───
        is_speaking = lod > FUSION_CFG.LOD_THRESHOLD

        # ─── Paso 6: Extraer coordenadas en píxeles para debug ───
        h, w = frame_bgr.shape[:2]
        landmarks_px: list[tuple[int, int]] = []
        for idx in TODOS_LABIOS_INDICES:
            # La nueva API devuelve una lista de NormalizedLandmark
            # Accedemos por índice directo
            lm = face_landmarks[idx]
            px = int(lm.x * w)
            py = int(lm.y * h)
            landmarks_px.append((px, py))

        state = LipState(
            timestamp=time.time(),
            lod=lod,
            is_speaking=is_speaking,
            landmarks_pixels=landmarks_px,
        )

        # Agregar al buffer temporal
        self._lip_states.append(state)
        self._last_frame_bgr = frame_bgr

        # ─── Paso 7: Generar frame de debug ───
        self._generate_debug_frame(frame_bgr, state, face_landmarks)

        return state

    def _calculate_lod(self, face_landmarks) -> float:
        """
        Calcula la Lip Opening Distance (LOD) a partir de los landmarks.

        Fórmula:
            LOD = (1/N) × Σ d(L_superior_i, L_inferior_i)

        Donde:
            d(a, b) = √[(ax - bx)² + (ay - by)²]
            N = número de pares (3 en nuestro caso)

        Args:
            face_landmarks: Lista de NormalizedLandmark de MediaPipe (478 puntos).

        Returns:
            LOD como float en coordenadas normalizadas [0, ~0.15].
            Valores típicos:
              - Boca cerrada: 0.001 - 0.010
              - Boca ligeramente abierta: 0.015 - 0.030
              - Boca muy abierta (vocal "A"): 0.040 - 0.080
        """
        distancias: list[float] = []

        for idx_superior, idx_inferior in PARES_LOD:
            # Obtener landmarks del par (acceso por índice en la lista)
            lm_sup = face_landmarks[idx_superior]
            lm_inf = face_landmarks[idx_inferior]

            # Distancia euclidiana 2D (ignoramos Z para simplificar).
            # Z de MediaPipe es una profundidad relativa, no es confiable
            # para medir apertura labial en el plano frontal.
            dx = lm_sup.x - lm_inf.x
            dy = lm_sup.y - lm_inf.y
            distancia = math.sqrt(dx * dx + dy * dy)
            distancias.append(distancia)

        # LOD = promedio de las distancias de los 3 pares
        lod = sum(distancias) / len(distancias) if distancias else 0.0

        return lod

    def _generate_debug_frame(
        self, frame_bgr: np.ndarray, state: LipState, landmarks
    ) -> None:
        """
        Genera un frame de debug anotado con:
          1. Frame completo con landmarks de labios y estado.
          2. Recorte ampliado de la región labial incrustado.

        El resultado se codifica a JPEG y se guarda en self.latest_debug_jpeg.
        """
        debug_frame = frame_bgr.copy()
        h, w = debug_frame.shape[:2]

        if landmarks is not None and state.landmarks_pixels:
            color = (0, 255, 0) if state.is_speaking else (0, 0, 255)
            for px, py in state.landmarks_pixels:
                cv2.circle(debug_frame, (px, py), 1, color, -1)

            xs = [p[0] for p in state.landmarks_pixels]
            ys = [p[1] for p in state.landmarks_pixels]
            margin_x = int((max(xs) - min(xs)) * 0.3)
            margin_y = int((max(ys) - min(ys)) * 0.3)
            x1 = max(0, min(xs) - margin_x)
            y1 = max(0, min(ys) - margin_y)
            x2 = min(w, max(xs) + margin_x)
            y2 = min(h, max(ys) + margin_y)

            if x2 > x1 and y2 > y1:
                lip_crop = debug_frame[y1:y2, x1:x2]
                crop_h, crop_w = lip_crop.shape[:2]
                scaled = cv2.resize(
                    lip_crop,
                    (
                        crop_w * DEBUG_CFG.LIP_CROP_SCALE,
                        crop_h * DEBUG_CFG.LIP_CROP_SCALE,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
                self._last_lip_crop = scaled
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)

        estado_txt = "HABLANDO" if state.is_speaking else "SILENCIO"
        color_txt = (0, 255, 0) if state.is_speaking else (0, 0, 255)

        cv2.putText(
            debug_frame,
            f"Estado: {estado_txt} | LOD: {state.lod:.4f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_txt,
            2,
        )

        cv2.putText(
            debug_frame,
            f"Umbral: {FUSION_CFG.LOD_THRESHOLD:.4f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        if self._last_lip_crop is not None:
            crop_h, crop_w = self._last_lip_crop.shape[:2]
            if crop_h < h and crop_w < w:
                pad = 2
                x_offset = w - crop_w - 10
                y_offset = 10
                cv2.rectangle(
                    debug_frame,
                    (x_offset - pad, y_offset - pad),
                    (x_offset + crop_w + pad, y_offset + crop_h + pad),
                    (255, 255, 255),
                    2,
                )
                debug_frame[
                    y_offset : y_offset + crop_h, x_offset : x_offset + crop_w
                ] = self._last_lip_crop

        # Codificar a JPEG (calidad 60 para que sea ligero al enviar por WebSocket)
        success, jpeg_buffer = cv2.imencode(
            ".jpg", debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 60]
        )
        if success:
            self.latest_debug_jpeg = jpeg_buffer.tobytes()
        else:
            self.latest_debug_jpeg = None

    def get_states_in_window(self, t_start: float, t_end: float) -> list[LipState]:
        """
        Devuelve los estados labiales dentro de una ventana temporal.

        Usado por el FusionEngine para correlacionar con la transcripción
        de audio del mismo intervalo de tiempo.

        Args:
            t_start: Timestamp de inicio de la ventana (time.time()).
            t_end: Timestamp de fin de la ventana.

        Returns:
            Lista de LipState cuyos timestamps están en [t_start, t_end].
        """
        return [
            state for state in self._lip_states if t_start <= state.timestamp <= t_end
        ]

    def get_speaking_ratio(self, t_start: float, t_end: float) -> float:
        """
        Calcula el ratio de frames con labios abiertos en una ventana temporal.

        Fórmula:
            speaking_ratio = N_hablando / N_total

        Args:
            t_start: Timestamp de inicio.
            t_end: Timestamp de fin.

        Returns:
            Ratio en [0.0, 1.0]. Si no hay estados en la ventana, devuelve 0.0.
        """
        states = self.get_states_in_window(t_start, t_end)

        if not states:
            return 0.0

        speaking_count = sum(1 for s in states if s.is_speaking)
        return speaking_count / len(states)

    def get_average_lod(self, t_start: float, t_end: float) -> float:
        """
        Calcula el LOD promedio en una ventana temporal.

        Args:
            t_start: Timestamp de inicio.
            t_end: Timestamp de fin.

        Returns:
            LOD promedio, o 0.0 si no hay estados en la ventana.
        """
        states = self.get_states_in_window(t_start, t_end)

        if not states:
            return 0.0

        return sum(s.lod for s in states) / len(states)

    @property
    def latest_state(self) -> Optional[LipState]:
        """Devuelve el último estado labial registrado."""
        return self._lip_states[-1] if self._lip_states else None

    def reset(self) -> None:
        """Limpia el buffer de estados y cierra ventanas de debug."""
        self._lip_states.clear()
        self._last_frame_bgr = None
        self._last_lip_crop = None
        logger.debug("VideoProcessor reseteado.")

    def cleanup(self) -> None:
        """Libera recursos visuales. Llamar al desconectar el WebSocket."""
        try:
            cv2.destroyAllWindows()
            # Forzar procesamiento de eventos pendientes de OpenCV
            for _ in range(5):
                cv2.waitKey(1)
            logger.info("Ventanas de OpenCV cerradas correctamente.")
        except Exception as e:
            logger.warning("Error al cerrar ventanas de OpenCV: %s", str(e))
