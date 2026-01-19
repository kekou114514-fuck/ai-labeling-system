# 使用支持 OpenGL 的 PyTorch 基础镜像
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 设置时区和语言
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 🔥 [加速] 替换 apt 为清华源
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 1. 安装系统级依赖
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

# 🔥 [加速] 配置 pip 默认使用清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 分步安装 Python 依赖 (防止冲突 & 锁定版本)

# [Step A] 升级 pip 并 锁定 NumPy 版本 (关键修复!)
# 先卸载可能存在的冲突版本，再安装稳定的 1.26.4
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y numpy && \
    pip install --no-cache-dir "numpy==1.26.4"

# [Step B] 安装核心库 & 视觉库 (YOLO)
# 提前安装 opencv-headless 防止 ultralytics 拉取带 GUI 的版本
RUN pip install --no-cache-dir \
    label-studio-sdk \
    opencv-python-headless \
    ultralytics \
    psycopg2-binary \
    pydantic \
    fastapi \
    uvicorn

# [Step C] 安装音频 & NLP 库 (单独安装，避免与 YOLO 冲突)
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    evaluate \
    jiwer \
    accelerate \
    librosa

# 4. 预创建数据挂载点
RUN mkdir -p /data
