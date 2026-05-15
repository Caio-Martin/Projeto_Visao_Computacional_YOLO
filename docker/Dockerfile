FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for OpenCV, Tkinter, and video support
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tk \
        tcl \
        ffmpeg \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
        opencv-python \
        numpy \
        matplotlib \
        ultralytics \
        Pillow

COPY . /app

CMD ["python", "sistema_visao_ia.py", "gui"]
