FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /workspace

# System dependencies (needed if you later build OpenCV with CUDA)
RUN apt-get update && apt-get install -y \
    cmake build-essential git pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements_gpu.txt .

# Remove CPU-only OpenCV (important)
RUN sed -i '/opencv-python-headless/d' requirements_gpu.txt
RUN sed -i 's/mediapipe==0.10.21/mediapipe==0.10.18/' requirements_gpu.txt
# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements_gpu.txt

# Copy project code
COPY . .

CMD ["bash"]
