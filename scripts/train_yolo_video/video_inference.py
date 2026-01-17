import os
import glob
import json
import sys
from ultralytics import YOLO

# 🚨 强制开启实时日志
sys.stdout.reconfigure(line_buffering=True)

# ==========================================
# ⚙️ Docker 适配配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.getenv('DATA_ROOT', '/data')  # 从环境变量读取

# 1. 图片路径 (对应 project_data/video_frames)
IMAGE_FOLDER = os.path.join(DATA_ROOT, "video_frames")

# 2. 输出文件 (建议放在 outputs 目录)
OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "pre_annotations_video.json")

# 3. 标签映射
LABELS_MAP = {0: "defect", 1: "scratch"}

# 4. Label Studio 本地文件前缀
LS_URL_PREFIX = "/data/local-files/?d=/data/"
# ==========================================

def get_best_model():
    """自动寻找最佳模型"""
    # 优先找 Docker 里的训练结果
    candidates = glob.glob(os.path.join(BASE_DIR, "run_video_v*/weights/best.pt"))
    if not candidates: return None
    return max(candidates, key=os.path.getmtime)

def run_inference():
    print("-" * 40)
    print("🎬 启动视频专用推理 (Docker版)")
    print("-" * 40)

    # 1. 加载模型
    model_path = get_best_model()
    if model_path:
        print(f"✅ 使用训练模型: {os.path.relpath(model_path, BASE_DIR)}")
        model = YOLO(model_path)
    else:
        # 如果没训练过，尝试使用预置的基础模型
        fallback = "/app/models/yolov8n.pt"
        if os.path.exists(fallback):
            print(f"⚠️ 使用基础模型: {fallback}")
            model = YOLO(fallback)
        else:
            print("⚠️ 下载官方 yolov8n.pt...")
            model = YOLO('yolov8n.pt')

    # 2. 扫描图片
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 错误：找不到图片目录 {IMAGE_FOLDER}")
        return

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
        image_files.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext.upper())))

    if not image_files:
        print(f"❌ 目录为空: {IMAGE_FOLDER}")
        return

    print(f"🖼️  正在处理 {len(image_files)} 张图片...")

    # 3. 执行推理
    results_list = []
    for i, img_path in enumerate(image_files):
        try:
            results = model(img_path, conf=0.25, verbose=False)
        except: continue
        
        predictions = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label_name = LABELS_MAP.get(cls_id)
                if not label_name: continue

                x, y, w, h = box.xywhn[0].tolist()
                predictions.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": (x - w / 2) * 100, "y": (y - h / 2) * 100,
                        "width": w * 100, "height": h * 100,
                        "rectanglelabels": [label_name]
                    },
                    "score": float(box.conf[0])
                })

        # 生成 Docker 兼容的 URL
        # 物理路径: /data/video_frames/1.jpg
        # URL: /data/local-files/?d=/data/video_frames/1.jpg
        rel_path = os.path.relpath(img_path, DATA_ROOT)
        ls_url = f"{LS_URL_PREFIX}{rel_path}"

        results_list.append({
            "data": {"image": ls_url},
            "predictions": [{"model_version": "yolo_video_v1", "score": 0.5, "result": predictions}]
        })
        
        if (i + 1) % 10 == 0: print(f"   已处理 {i + 1}/{len(image_files)}...")

    # 4. 保存
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)

    print(f"🎉 推理完成！结果已保存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    run_inference()
