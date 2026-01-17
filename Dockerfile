# 使用支持 OpenGL 的 PyTorch 基础镜像 (为了 OpenCV 和 Whisper)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 设置时区和语言
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 1. 安装系统级依赖 (FFmpeg 用于 Whisper, LibGL 用于 OpenCV)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. 设置工作目录
WORKDIR /app

# 3. 安装 Python 依赖
# 咱们把 requirements 直接写在这里，减少文件数量
RUN pip install --no-cache-dir \
    label-studio-sdk \
    ultralytics \
    transformers \
    datasets \
    evaluate \
    jiwer \
    accelerate \
    librosa \
    opencv-python-headless \
    pydantic \
    fastapi \
    uvicorn

# 4. 预创建数据挂载点
RUN mkdir -p /data
