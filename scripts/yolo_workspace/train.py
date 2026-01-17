import os
import sys
import torch
from ultralytics import YOLO

# 强制开启日志
sys.stdout.reconfigure(line_buffering=True)

# 1. 设备检测
if torch.cuda.is_available():
    DEVICE = '0'
    print(f"🚀 GPU 模式: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'
    print("⚠️ CPU 模式")

# 2. 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(BASE_DIR, 'data.yaml')
PROJECT_DIR = os.path.join(BASE_DIR, 'runs/detect')

# 3. 加载模型 (优先使用 Docker 映射的离线模型)
LOCAL_MODEL = "/app/models/yolov8n.pt"
if os.path.exists(LOCAL_MODEL):
    print(f"📥 加载离线模型: {LOCAL_MODEL}")
    model = YOLO(LOCAL_MODEL)
else:
    print("⚠️ 未找到离线模型，尝试下载...")
    model = YOLO('yolov8n.pt') 

# 4. 开始训练
print(f"🚀 读取配置: {YAML_PATH}")
try:
    results = model.train(
        data=YAML_PATH,
        epochs=100,
        imgsz=640,
        batch=8,
        device=DEVICE,
        project=PROJECT_DIR,
        name='my_defect_project', 
        exist_ok=True
    )
    print("🎉 P1 训练成功！")
except Exception as e:
    print(f"❌ 训练失败: {e}")
    sys.exit(1)
