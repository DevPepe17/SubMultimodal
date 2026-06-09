# AVSR — Reconocimiento del Habla Audio-Visual (Audio-Visual Speech Recognition)

Este proyecto implementa un sistema avanzado de **Reconocimiento del Habla Audio-Visual en Tiempo Real**. Su objetivo principal es resolver el problema de las "alucinaciones" (falsos positivos donde la IA transcribe texto sin que el usuario haya hablado), combinando la potencia de **Faster-Whisper** para el procesamiento de audio con **MediaPipe FaceLandmarker** para el análisis visual del movimiento de los labios.

---

## 🚀 Arquitectura del Sistema

El sistema utiliza una arquitectura cliente-servidor basada en **WebSockets (Full-Duplex)** y **Fusión Tardía (Late Fusion)**. 

1. **Cliente Web:** Captura el micrófono y la cámara del navegador, y envía ambos flujos (multiplexados de forma binaria) al servidor.
2. **Servidor FastAPI:** Separa los flujos. El video se procesa en tiempo real (30 FPS) para medir la apertura de los labios, mientras que el audio se almacena en un buffer y se transcribe cada 3 segundos.
3. **Motor de Fusión (Fusion Engine):** Cuando Whisper devuelve un texto, el motor revisa el historial de video de esos últimos 3 segundos. Si la persona tuvo la boca cerrada la mayor parte del tiempo, se descarta el texto asumiéndolo como una alucinación o ruido de fondo.

---

## 📂 Estructura del Proyecto y Archivos

### 1. `app/main.py`
**El corazón del servidor.** 
- Configura la aplicación **FastAPI**.
- Utiliza el patrón `lifespan` para precargar en la memoria RAM los modelos pesados de IA (Whisper y MediaPipe) al momento de encender el servidor, garantizando que no haya latencia en la primera conexión.
- Sirve la página web principal y enlaza el router del WebSocket.

### 2. `app/api/websocket.py`
**El controlador de streaming en tiempo real.**
- Expone el endpoint `/stream`.
- Recibe mensajes binarios prefijados: `0x01` para Audio y `0x02` para Video.
- **Concurrencia asíncrona:** Ejecuta el análisis de audio de Whisper en una tarea en segundo plano (`asyncio.create_task`) para no bloquear la recepción de los frames de video de la cámara, logrando una fluidez visual perfecta de 30 FPS.
- Envía de vuelta al cliente los resultados de la transcripción en JSON y los frames de video de depuración (`0x03`).

### 3. `app/core/config.py`
**Configuración centralizada.**
- Contiene todas las constantes del sistema como la frecuencia de muestreo de audio (16kHz), umbrales de confianza, tiempos de buffer, y el **`LOD_THRESHOLD`** (Umbral de Apertura Labial) crucial para decidir si la boca está abierta o cerrada.

### 4. `app/models_ia/whisper_model.py`
**Envoltura del Modelo de Audio.**
- Implementa un patrón **Singleton** para cargar `faster-whisper` (modelo `small` en formato `int8` optimizado para CPU). 
- Expone el método para transcribir arreglos de audio y obtener tanto el texto como la probabilidad logarítmica (`avg_logprob`).

### 5. `app/models_ia/lip_reading_model.py`
**Envoltura del Modelo Visual.**
- Implementa un patrón **Singleton** para la nueva Tasks API de **MediaPipe FaceLandmarker**.
- Recibe frames RGB y devuelve las coordenadas espaciales de los 478 puntos (landmarks) del rostro.

### 6. `app/services/video_processor.py`
**Procesador de Visión Computacional.**
- Recibe los bytes JPEG de la cámara web, los decodifica y se los pasa a MediaPipe.
- Extrae los puntos específicos de los labios y calcula el **LOD (Lip Opening Distance)** promediando la distancia euclidiana entre 3 pares de puntos labiales (superior e inferior).
- Dibuja los puntos verdes/rojos sobre el rostro del usuario, hace un recorte (crop) ampliado de los labios, codifica este "frame de debug" a JPEG y lo guarda para enviarlo de vuelta a la web.

### 7. `app/services/audio_processor.py`
**Procesador de Señal Acústica.**
- Mantiene un búfer circular de bytes de audio PCM.
- Cuando acumula 3 segundos de audio, convierte los enteros de 16 bits en un arreglo `float32` normalizado.
- Envía este arreglo a Whisper utilizando un `ThreadPoolExecutor` para no bloquear el hilo principal (Event Loop) de Python.

### 8. `app/services/fusion_engine.py`
**El Árbitro Final.**
- Recibe un objeto `AudioResult` con texto de Whisper.
- Le pide al `VideoProcessor` los estados labiales históricos que coincidan con la marca de tiempo de ese audio.
- Calcula el `speaking_ratio` (porcentaje del tiempo en que el LOD superó el umbral).
- Toma la decisión final: **Aprobada** (la persona habló), **Penalizada** (baja confianza o ratio dudoso), o **Descartada** (los labios estuvieron cerrados).

### 9. `app/static/index.html`
**La Interfaz de Usuario (Cliente Web).**
- Una web moderna y responsiva construida con Vanilla JS, HTML y CSS avanzado (Glassmorphism).
- Utiliza la API del navegador `navigator.mediaDevices.getUserMedia` para acceder a la webcam y el micrófono localmente.
- Dibuja el video en un `<canvas>` interno, lo codifica a JPEG a 30 FPS y captura el audio del micrófono pasándolo por un nodo `ScriptProcessor` a 16kHz.
- Recibe y muestra el video procesado por el servidor superpuesto en la interfaz, además de mostrar el historial dinámico de transcripciones con alertas de colores según el veredicto del motor de fusión.

### 10. Archivos adicionales
- `models_weight/face_landmarker.task`: Los pesos descargados requeridos por MediaPipe.
- `test/test_client.py`: Cliente original hecho en Python + OpenCV, usado para desarrollo antes de crear la interfaz web.

---

## ⚙️ Cómo funciona el cálculo de LOD (Lip Opening Distance)
El sistema evita depender de hardware complejo midiendo la distancia en el plano 2D frontal entre los labios superior e inferior:
1. Extrae las coordenadas de 3 pares de puntos labiales (centro, izquierda, derecha).
2. Calcula la distancia euclidiana entre cada par.
3. El promedio de estas 3 distancias es el **LOD**.
4. Si `LOD > 0.0150` (configurable), se considera que la boca está abierta o articulando (Estado: HABLANDO). En caso contrario, Estado: SILENCIO.

---
*Desarrollado como una arquitectura tolerante a falsos positivos de Inteligencia Artificial Generativa Acústica.*
