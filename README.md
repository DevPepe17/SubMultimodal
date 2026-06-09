# 🎙️ AVSR: Reconocimiento del Habla Audio-Visual en Tiempo Real

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceLandmarker-orange.svg)
![Faster-Whisper](https://img.shields.io/badge/Faster--Whisper-Systran-purple.svg)

**AVSR** (Audio-Visual Speech Recognition) es un sistema avanzado diseñado para solucionar uno de los problemas más comunes en los modelos de Inteligencia Artificial de voz: **las alucinaciones acústicas**.

Los modelos como Whisper a menudo generan texto ("alucinaciones") cuando hay ruido de fondo o silencios prolongados, asumiendo erróneamente que alguien está hablando. Este proyecto soluciona esto implementando una **Fusión Tardía (Late Fusion)**: combina el audio en tiempo real con el seguimiento visual de los labios a través de la cámara web. Si el modelo detecta "palabras" pero tus labios están cerrados, el texto se descarta automáticamente.

## ✨ Características Principales

- **Fusión Audio-Visual:** Combina el análisis acústico (Faster-Whisper) con la visión computacional (MediaPipe) para validar si realmente estás hablando.
- **Interfaz Web Moderna:** Cliente frontend en el navegador con acceso a cámara/micrófono, diseño *glassmorphism* y modo oscuro.
- **Streaming por WebSockets:** Transmisión multiplexada de audio PCM y video JPEG de altísima velocidad sin latencia perceptible.
- **Procesamiento Asíncrono:** La interfaz visual corre a 30 FPS fluidos mientras la transcripción de IA ocurre en segundo plano de manera no bloqueante.
- **100% Local y Privado:** Todo el procesamiento (Visión e Inteligencia Artificial) se ejecuta en tu propia máquina. Ningún dato se envía a la nube.

## 🛠️ Tecnologías

- **Backend:** Python, FastAPI, WebSockets, Uvicorn, OpenCV.
- **IA de Audio:** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (modelo `small` optimizado en int8).
- **IA de Visión:** [MediaPipe FaceLandmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) (API de Tareas).
- **Frontend:** Vanilla JavaScript, HTML5, CSS3, `getUserMedia` API.

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/audio_visual_asr.git
cd audio_visual_asr
```

### 2. Crear un entorno virtual e instalar dependencias
Requiere Python 3.10 o superior.
```bash
# Crear entorno
python -m venv venv

# Activar (Windows)
venv\Scripts\Activate.ps1
# Activar (Linux/Mac)
source venv/bin/activate

# Instalar librerías
pip install -r requirements.txt
```

### 3. Ejecutar el servidor
El sistema iniciará la descarga automática del modelo Whisper (la primera vez) y levantará el servidor local.
```bash
python -m app.main
```

### 4. Abrir la Aplicación
1. Abre tu navegador (Chrome, Edge o Firefox).
2. Entra a [http://localhost:8000](http://localhost:8000).
3. Haz clic en **"Iniciar"** y concede los permisos de cámara y micrófono.
4. ¡Habla a la cámara! Podrás ver los puntos de rastreo labial sobre tu rostro y las transcripciones filtradas en tiempo real.

---

## 🧠 ¿Cómo funciona internamente?

1. **Calculo del LOD (Lip Opening Distance):** El servidor usa MediaPipe para encontrar tu rostro y calcular la distancia promedio entre 3 pares de puntos específicos de tus labios.
2. **Umbral de Decisión:** Si el LOD supera `0.0150`, la boca está abierta o articulando. Si es menor, está en silencio.
3. **Fusión:** Cuando Whisper devuelve una frase, se revisa el historial del LOD de esos mismos segundos. Si estuviste en silencio, la frase se descarta como "Alucinación filtrada".

> 📘 **¿Eres desarrollador?**
> Para entender a fondo la arquitectura, el flujo de datos asíncrono y los detalles de cada archivo del proyecto, consulta la [Guía del Desarrollador (GUIA_DESARROLLADOR.md)](./GUIA_DESARROLLADOR.md).

---
*Desarrollado como una solución tolerante a fallos para Inteligencia Artificial Acústica.*
