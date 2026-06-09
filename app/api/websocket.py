"""
=============================================================================
 websocket.py — Controlador del WebSocket /stream
=============================================================================
 Punto de entrada principal para las conexiones WebSocket de streaming
 audio-visual en tiempo real.

 Protocolo de comunicación:
   - El cliente envía mensajes BINARIOS con un prefijo de 1 byte:
       0x01 = audio (PCM 16-bit, 16kHz, mono)
       0x02 = video (frame JPEG)
   - El servidor responde con mensajes TEXT (JSON) cuando hay una
     transcripción lista (después de pasar por el FusionEngine).

 Ciclo de vida de una conexión:
   1. CONNECT: Instanciar procesadores (Audio, Video, Fusion) para la sesión.
   2. LOOP: Recibir datos → despachar al procesador → enviar resultado.
   3. DISCONNECT: Capturar WebSocketDisconnect → limpiar recursos → cerrar.

 NOTA sobre concurrencia:
   Cada conexión WebSocket tiene sus PROPIAS instancias de AudioProcessor,
   VideoProcessor y FusionEngine. Los MODELOS de IA (Whisper, MediaPipe)
   son compartidos vía Singleton. Esto permite múltiples sesiones sin
   recargar los modelos (~500MB cada uno).
=============================================================================
"""

from __future__ import annotations

import json
import logging
from typing import Final

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.audio_processor import AudioProcessor
from app.services.fusion_engine import FusionEngine
from app.services.video_processor import VideoProcessor

# Logger del módulo
logger = logging.getLogger(__name__)

# Router de FastAPI para los endpoints de WebSocket
router = APIRouter()

# ─── Constantes del protocolo binario ───
# Prefijos de 1 byte para identificar el tipo de datos en el stream.
PREFIX_AUDIO: Final[int] = 0x01  # 1 = audio PCM
PREFIX_VIDEO: Final[int] = 0x02  # 2 = video JPEG


@router.websocket("/stream")
async def websocket_stream_endpoint(websocket: WebSocket) -> None:
    """
    Endpoint WebSocket para streaming audio-visual en tiempo real.

    Recibe flujos multiplexados de audio y video, procesa cada modalidad
    de forma independiente, y aplica fusión tardía para producir
    transcripciones filtradas.

    Args:
        websocket: Conexión WebSocket de FastAPI.
    """
    client_id = (
        f"{websocket.client.host}:{websocket.client.port}"
        if websocket.client
        else "desconocido"
    )
    logger.info("═══ Nueva conexión WebSocket desde %s ═══", client_id)

    # ─── Aceptar la conexión ───
    await websocket.accept()
    logger.info("✓ Conexión WebSocket aceptada para cliente %s", client_id)

    # ─── Instanciar procesadores para esta sesión ───
    audio_processor = AudioProcessor()
    video_processor = VideoProcessor()
    fusion_engine = FusionEngine(video_processor)

    try:
        # ─── Loop principal de recepción ───
        while True:
            # Recibir mensaje binario del cliente.
            # Si el cliente se desconecta, esto lanza WebSocketDisconnect.
            data: bytes = await websocket.receive_bytes()

            # Verificar que hay al menos 1 byte de prefijo + payload
            if len(data) < 2:
                logger.warning(
                    "Mensaje demasiado corto (%d bytes) de %s. Ignorando.",
                    len(data),
                    client_id,
                )
                continue

            # ─── Leer el byte de prefijo ───
            prefix: int = data[0]
            payload: bytes = data[1:]

            # ─── Despachar según el tipo de dato ───
            if prefix == PREFIX_AUDIO:
                # ─── Procesar audio ───
                audio_processor.add_audio_chunk(payload)

                # Definir la tarea asíncrona para no bloquear el loop de video
                async def process_and_send_audio():
                    # Intentar transcribir si hay suficiente audio acumulado
                    audio_result = await audio_processor.process_if_ready()

                    if audio_result is not None and audio_result.tiene_texto:
                        # Pasar por el motor de fusión
                        fusion_result = fusion_engine.evaluate(audio_result)

                        # Enviar resultado al cliente como JSON
                        try:
                            await websocket.send_text(
                                json.dumps(fusion_result.to_dict(), ensure_ascii=False)
                            )
                        except Exception:
                            pass

                # Ejecutar en el background para seguir recibiendo frames de video
                import asyncio

                asyncio.create_task(process_and_send_audio())

            elif prefix == PREFIX_VIDEO:
                # ─── Procesar video ───
                # El procesamiento de video actualiza el buffer interno de
                # estados labiales. No produce una respuesta directa al cliente,
                # sino que alimenta al FusionEngine para las próximas decisiones.
                video_processor.process_frame_bytes(payload)

                # Si se generó el frame de debug anotado, enviarlo de vuelta
                if video_processor.latest_debug_jpeg:
                    # Usamos 0x03 como prefijo para "Video de Debug"
                    msg = bytes([0x03]) + video_processor.latest_debug_jpeg
                    try:
                        await websocket.send_bytes(msg)
                    except Exception:
                        pass

            else:
                logger.warning(
                    "Prefijo desconocido 0x%02X de %s. Ignorando mensaje.",
                    prefix,
                    client_id,
                )

    except WebSocketDisconnect:
        # ─── Desconexión limpia del cliente ───
        logger.info("Cliente %s desconectado del WebSocket.", client_id)

    except Exception as e:
        # ─── Error inesperado ───
        logger.error(
            "Error inesperado en WebSocket de %s: %s",
            client_id,
            str(e),
            exc_info=True,
        )
        # Intentar notificar al cliente antes de cerrar
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "tipo": "error",
                        "mensaje": f"Error interno del servidor: {str(e)}",
                    }
                )
            )
        except Exception:
            pass  # El WebSocket puede estar ya cerrado

    finally:
        # ─── Limpieza de recursos (SIEMPRE se ejecuta) ───
        logger.info("Limpiando recursos de la sesión %s...", client_id)

        # Resetear procesadores
        audio_processor.reset()
        video_processor.reset()
        fusion_engine.reset()

        # Cerrar ventanas de OpenCV (CRÍTICO para evitar que el servidor se cuelgue).
        # cv2.destroyAllWindows() debe llamarse desde el hilo que creó las ventanas.
        video_processor.cleanup()

        # Log de estadísticas de la sesión
        stats = fusion_engine.stats
        logger.info(
            "═══ Sesión %s finalizada ═══ "
            "aprobadas=%d | penalizadas=%d | descartadas=%d | vacías=%d",
            client_id,
            stats["aprobadas"],
            stats["penalizadas"],
            stats["descartadas"],
            stats["vacias"],
        )
