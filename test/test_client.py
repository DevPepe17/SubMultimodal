"""
=============================================================================
 test_client.py — Cliente de Streaming en Vivo para el Servidor AVSR
=============================================================================
 IMPORTANTE (Windows):
   cv2.imshow() SOLO funciona en el hilo principal del proceso.
   Por eso la arquitectura de hilos es:

   - HILO PRINCIPAL: captura de video + cv2.imshow + cv2.waitKey
   - Hilo secundario 1: captura de audio con PyAudio
   - Hilo secundario 2: event loop asyncio con WebSocket (envío/recepción)

 Uso:
   python test/test_client.py

 Controles:
   - Presiona 'q' en la ventana de video para cerrar.
   - Ctrl+C en la terminal tambien cierra todo limpiamente.
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import threading
import time
from queue import Empty, Queue
from typing import Optional

import cv2

# ─────────────────────────────────────────────────────────────────────────────
# Configuracion del cliente
# ─────────────────────────────────────────────────────────────────────────────
SERVER_URL: str = "ws://localhost:8000/stream"

SAMPLE_RATE: int = 16_000
CHANNELS: int = 1
SAMPLE_WIDTH: int = 2
AUDIO_CHUNK_SAMPLES: int = 4096

CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
VIDEO_FPS: int = 15
JPEG_QUALITY: int = 70

PREFIX_AUDIO: int = 0x01
PREFIX_VIDEO: int = 0x02

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_client")

# ─────────────────────────────────────────────────────────────────────────────
# Estado global y colas thread-safe
# ─────────────────────────────────────────────────────────────────────────────
_running: bool = True

# Cola para audio capturado (hilo audio → hilo websocket)
_audio_queue: Queue[bytes] = Queue(maxsize=50)

# Cola para video capturado (hilo principal → hilo websocket)
_video_queue: Queue[bytes] = Queue(maxsize=10)

# Cola para respuestas del servidor (hilo websocket → hilo principal)
_response_queue: Queue[dict] = Queue(maxsize=50)


def stop_all() -> None:
    global _running
    _running = False


# ─────────────────────────────────────────────────────────────────────────────
# HILO DE AUDIO (secundario) — captura bloqueante con PyAudio
# ─────────────────────────────────────────────────────────────────────────────
def audio_capture_thread() -> None:
    """Captura audio del microfono y deposita chunks en _audio_queue."""
    import pyaudio

    pa: Optional[pyaudio.PyAudio] = None
    stream = None

    try:
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK_SAMPLES,
        )
        logger.info(
            "Microfono abierto: %d Hz, %d canal(es), %d-bit",
            SAMPLE_RATE,
            CHANNELS,
            SAMPLE_WIDTH * 8,
        )

        while _running:
            try:
                pcm_data: bytes = stream.read(
                    AUDIO_CHUNK_SAMPLES, exception_on_overflow=False
                )
                try:
                    _audio_queue.put_nowait(pcm_data)
                except Exception:
                    pass
            except Exception as e:
                if _running:
                    logger.error("Error captura audio: %s", str(e))
                break

    except Exception as e:
        logger.error("Error al inicializar PyAudio: %s", str(e))
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
        logger.info("Microfono cerrado.")


# ─────────────────────────────────────────────────────────────────────────────
# HILO DE WEBSOCKET (secundario) — asyncio event loop propio
# ─────────────────────────────────────────────────────────────────────────────
def websocket_thread() -> None:
    """
    Corre un event loop asyncio en un hilo secundario.
    Maneja la conexion WebSocket, envio de datos desde colas,
    y recepcion de respuestas del servidor.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_websocket_main())
    except Exception as e:
        if _running:
            logger.error("Error en hilo WebSocket: %s", str(e))
    finally:
        loop.close()


async def _websocket_main() -> None:
    """Conexion y comunicacion WebSocket."""
    import websockets

    try:
        async with websockets.connect(
            SERVER_URL,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=60,
        ) as ws:
            logger.info("Conectado al servidor AVSR.")

            send_task = asyncio.create_task(_send_loop(ws))
            recv_task = asyncio.create_task(_recv_loop(ws))

            # Esperar hasta que _running sea False
            while _running:
                await asyncio.sleep(0.1)

            send_task.cancel()
            recv_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

    except ConnectionRefusedError:
        logger.error(
            "No se pudo conectar a %s. Esta el servidor corriendo?", SERVER_URL
        )
        stop_all()
    except Exception as e:
        if _running:
            logger.error("Error WebSocket: %s", str(e))
        stop_all()


async def _send_loop(ws) -> None:
    """Lee datos de las colas y los envia por WebSocket."""
    while _running:
        sent = False

        # Enviar audio
        try:
            while not _audio_queue.empty():
                pcm = _audio_queue.get_nowait()
                await ws.send(bytes([PREFIX_AUDIO]) + pcm)
                sent = True
        except Empty:
            pass
        except Exception as e:
            if _running:
                logger.error("Error envio audio: %s", str(e))
            return

        # Enviar video
        try:
            while not _video_queue.empty():
                jpeg = _video_queue.get_nowait()
                await ws.send(bytes([PREFIX_VIDEO]) + jpeg)
                sent = True
        except Empty:
            pass
        except Exception as e:
            if _running:
                logger.error("Error envio video: %s", str(e))
            return

        if not sent:
            await asyncio.sleep(0.005)


async def _recv_loop(ws) -> None:
    """Recibe respuestas JSON del servidor y las pone en _response_queue."""
    while _running:
        try:
            response = await asyncio.wait_for(ws.recv(), timeout=0.5)
            data = json.loads(response)
            try:
                _response_queue.put_nowait(data)
            except Exception:
                pass
        except asyncio.TimeoutError:
            continue
        except json.JSONDecodeError:
            pass
        except Exception as e:
            if _running:
                logger.error("Error recepcion: %s", str(e))
            return


# ─────────────────────────────────────────────────────────────────────────────
# FUNCION para imprimir respuestas (llamada desde el hilo principal)
# ─────────────────────────────────────────────────────────────────────────────
def print_responses() -> None:
    """Procesa las respuestas acumuladas en _response_queue."""
    while not _response_queue.empty():
        try:
            data = _response_queue.get_nowait()
        except Empty:
            break

        tipo = data.get("tipo", "desconocido")

        if tipo == "transcripcion":
            fusion_ok = data.get("fusion_aprobada", False)
            texto = data.get("texto", "")
            motivo = data.get("motivo", "")
            speaking = data.get("speaking_ratio", 0)
            lod = data.get("lod_promedio", 0)

            if fusion_ok and texto:
                print(f"\n>> APROBADA | {texto}")
                print(
                    f"   ratio_habla={speaking:.2f} | LOD={lod:.4f} | motivo={motivo}"
                )
            elif texto:
                print(f"\n?? PENALIZADA | {texto}")
                print(
                    f"   ratio_habla={speaking:.2f} | LOD={lod:.4f} | motivo={motivo}"
                )
            else:
                if motivo and motivo != "transcripcion_vacia":
                    print("\nXX DESCARTADA | (alucinacion filtrada)")
                    print(
                        f"   ratio_habla={speaking:.2f} | LOD={lod:.4f} | motivo={motivo}"
                    )

        elif tipo == "error":
            logger.error("Error del servidor: %s", data.get("mensaje", "?"))


# ─────────────────────────────────────────────────────────────────────────────
# HILO PRINCIPAL — captura de video + cv2.imshow (DEBE ser main thread)
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Punto de entrada principal.

    El hilo principal se encarga de:
      1. Lanzar los hilos secundarios (audio, websocket).
      2. Capturar video de la camara con OpenCV.
      3. Mostrar la ventana de video con cv2.imshow.
      4. Detectar la tecla 'q' para cerrar.
      5. Imprimir respuestas del servidor.
    """
    logger.info("=" * 56)
    logger.info("  AVSR -- Cliente de Streaming en Vivo")
    logger.info("  Servidor: %s", SERVER_URL)
    logger.info("=" * 56)

    # ─── Lanzar hilos secundarios ───
    t_audio = threading.Thread(target=audio_capture_thread, daemon=True, name="audio")
    t_ws = threading.Thread(target=websocket_thread, daemon=True, name="websocket")

    t_audio.start()
    t_ws.start()

    # Dar tiempo al WebSocket para conectar
    time.sleep(1.0)

    # ─── Abrir camara en el hilo principal ───
    # CAP_DSHOW (DirectShow) evita que VideoCapture se cuelgue en Windows.
    # El backend por defecto (MSMF) puede bloquearse indefinidamente.
    logger.info("Abriendo camara (indice %d, backend DirectShow)...", CAMERA_INDEX)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        logger.error("No se pudo abrir la camara (indice %d).", CAMERA_INDEX)
        stop_all()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    logger.info("Camara abierta: %dx%d @ %d FPS", FRAME_WIDTH, FRAME_HEIGHT, VIDEO_FPS)
    logger.info("La ventana de debug se muestra en el SERVIDOR.")
    logger.info("Presiona Ctrl+C en esta terminal para salir.\n")

    frame_interval = 1.0 / VIDEO_FPS

    try:
        while _running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Codificar a JPEG y poner en cola para envio
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            success, jpeg_buffer = cv2.imencode(".jpg", frame, encode_params)
            if success:
                try:
                    _video_queue.put_nowait(jpeg_buffer.tobytes())
                except Exception:
                    pass

            # Procesar respuestas del servidor
            print_responses()

            # Controlar FPS
            time.sleep(frame_interval)

    except KeyboardInterrupt:
        logger.info("Ctrl+C detectado.")

    finally:
        stop_all()
        cap.release()
        logger.info("Camara cerrada.")

        # Esperar hilos
        t_audio.join(timeout=2)
        t_ws.join(timeout=2)
        logger.info("Cliente AVSR finalizado.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: stop_all())
    main()
