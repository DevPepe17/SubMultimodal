# Usamos una imagen oficial de Python ligera
FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para OpenCV (libGL)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear un usuario no root (Requisito de Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Establecer directorio de trabajo
WORKDIR /app

# Copiar y ejecutar dependencias primero (aprovechar caché de Docker)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY --chown=user . .

# Hugging Face Spaces requiere que la aplicación exponga el puerto 7860
ENV PORT=7860
EXPOSE 7860

# Comando para ejecutar el servidor
CMD ["python", "-m", "app.main"]
