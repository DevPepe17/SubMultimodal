"""
=============================================================================
 main.py — Punto de Entrada de FastAPI y Carga de Modelos
=============================================================================
 Este módulo configura la aplicación FastAPI, pre-carga los modelos de IA
 durante el arranque del servidor (usando el patrón lifespan), e incluye
 los routers de la API.

 Flujo de arranque:
   1. Se crea la app FastAPI con metadata del proyecto.
   2. Se registra el lifespan (contexto de vida de la app).
   3. Al arrancar (startup), se instancian los Singletons de Whisper y
      MediaPipe, cargándolos en RAM ANTES de aceptar conexiones.
   4. Al apagar (shutdown), se liberan recursos de MediaPipe.
   5. Se incluyen los routers (WebSocket /stream).
   6. Se configura el logging estructurado.
   7. Se lanza uvicorn si se ejecuta como script principal.

 Uso:
   python -m app.main
     o
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
=============================================================================
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.websocket import router as websocket_router
from app.models_ia.lip_reading_model import LipReadingModelSingleton
from app.models_ia.whisper_model import WhisperModelSingleton

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-35s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: Ciclo de vida de la aplicación
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Contexto de vida de la aplicación FastAPI.

    STARTUP (antes del yield):
      - Carga los modelos de IA en memoria RAM.
      - Esto bloquea temporalmente el arranque (~5-10s para Whisper),
        pero garantiza que la PRIMERA conexión WebSocket no tenga latencia
        de carga del modelo.

    SHUTDOWN (después del yield):
      - Libera recursos de MediaPipe de forma limpia.
      - Whisper (CTranslate2) libera memoria automáticamente vía
        destructor de C++.
    """
    # ═══════════ STARTUP ═══════════
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  AVSR — Sistema de Reconocimiento Audio-Visual     ║")
    logger.info("║  Iniciando carga de modelos de IA...               ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    try:
        # Cargar Faster-Whisper (el más pesado, ~5-10 segundos).
        logger.info("Paso 1/2: Cargando Faster-Whisper (modelo 'small', int8)...")
        whisper_singleton = WhisperModelSingleton()
        assert whisper_singleton.is_loaded, "Whisper no se cargó correctamente."
        logger.info("✓ Faster-Whisper listo.")

        # Cargar MediaPipe Face Mesh (ligero, <1 segundo).
        logger.info("Paso 2/2: Cargando MediaPipe Face Mesh...")
        mediapipe_singleton = LipReadingModelSingleton()
        assert mediapipe_singleton.is_loaded, "MediaPipe no se cargó correctamente."
        logger.info("✓ MediaPipe Face Mesh listo.")

        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║  ✓ Todos los modelos cargados exitosamente.         ║")
        logger.info("║  Servidor listo para recibir conexiones WebSocket.  ║")
        logger.info("╚══════════════════════════════════════════════════════╝")

    except Exception as e:
        logger.critical(
            "✗ Error FATAL al cargar modelos de IA: %s", str(e), exc_info=True
        )
        raise

    # ─── Yield: el servidor está activo y atendiendo requests ───
    yield

    # ═══════════ SHUTDOWN ═══════════
    logger.info("Apagando servidor AVSR... Liberando recursos.")

    try:
        mediapipe_singleton.close()
    except Exception as e:
        logger.warning("Error al liberar MediaPipe: %s", str(e))

    logger.info("Servidor AVSR apagado correctamente. ¡Hasta pronto!")


# ─────────────────────────────────────────────────────────────────────────────
# Creación de la aplicación FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AVSR — Reconocimiento del Habla Audio-Visual",
    description=(
        "Sistema de transcripción de voz en tiempo real con validación "
        "visual mediante seguimiento de labios. Utiliza Faster-Whisper "
        "para transcripción local en español y MediaPipe Face Mesh para "
        "detectar actividad labial y filtrar alucinaciones."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── Incluir routers ───
app.include_router(websocket_router)


# ─── Servir la Web App ───
@app.get("/")
async def get_index() -> FileResponse:
    """Sirve la página web del cliente AVSR."""
    index_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index_path)


# ─── Endpoint de salud (health check) ───
@app.get("/health")
async def health_check() -> dict:
    """Verifica que el servidor y los modelos estén operativos."""
    whisper_ok = WhisperModelSingleton().is_loaded
    mediapipe_ok = LipReadingModelSingleton().is_loaded

    return {
        "status": "ok" if (whisper_ok and mediapipe_ok) else "degraded",
        "modelos": {
            "whisper": "cargado" if whisper_ok else "error",
            "mediapipe": "cargado" if mediapipe_ok else "error",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada directo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Desactivar en producción para estabilidad
        log_level="info",
    )
