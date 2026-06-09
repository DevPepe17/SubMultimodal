"""Paquete api — Endpoints del sistema AVSR."""

from app.api.websocket import router as websocket_router

__all__: list[str] = [
    "websocket_router",
]
