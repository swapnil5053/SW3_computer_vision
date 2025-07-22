FROM tensorflow/tensorflow:2.15.0-gpu

# Basic system dependencies
RUN apt-get update && \
    apt-get install -y \
    sudo \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install opencv-python scipy numpy==1.26.4 \
    imageio IPython scikit-learn tqdm \
    fastapi uvicorn python-multipart requests pillow 

RUN pip install pymongo GPUtil psutil

RUN mkdir -p /workspace
WORKDIR /workspace

COPY ../ /workspace

CMD ["python", "/workspace/app.py"]
