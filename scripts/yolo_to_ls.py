import os
import glob
import json
import argparse
from ultralytics import YOLO

# ==========================================
# ⚙️ Docker 适配配置
# ==========================================
# 容器内数据根目录 (映射宿主机的 project_data)
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
# Label Studio 本地文件访问前缀
LS_URL_PREFIX = "/data/local-files/?d=/data/"
# 基础模型路径 (离线)
BASE_MODEL_PATH = "/app/models/yolov8n.pt"

def run_inference(project_type):
    # === P1: 产品图片 ===
    if project_type == '1':
        print("📦 模式: 项目 1 (产品图片)")
        config = {
            "images": os.path.join(DATA_ROOT, "images"),
            # 优先用训练好的最佳模型，如果没有则用基础模型
            "model": os.path.join(DATA_ROOT, "outputs/my_best_model.pt"),
            "output": os.path.join(DATA_ROOT, "outputs/pre_annotations_images.json"),
            "labels": {0: "物体框(Box)", 1: "文字区域", 2: "复杂轮廓(Poly)"}
        }
    # === P4: 视频抽帧 ===
    elif project_type == '4':
        print("🎬 模式: 项目 4 (视频抽帧图片)")
        config = {
            "images": os.path.join(DATA_ROOT, "video_frames"),
            # P4 暂时使用基础模型演示，或者您可以指定 train_yolo_video 跑出来的 best.pt
            "model": BASE_MODEL_PATH,
            "output": os.path.join(DATA_ROOT, "outputs/pre_annotations_video_frames.json"),
            "labels": {0: "defect", 1: "scratch"}
        }
    else:
        print(f"❌ 未知项目类型: {project_type}")
        return

    # 1. 检查模型
    if not os.path.exists(config['model']):
        print(f"⚠️ 指定模型不存在: {config['model']}")
        if os.path.exists(BASE_MODEL_PATH):
            print(f"🔄 自动切换为基础模型: {BASE_MODEL_PATH}")
            config['model'] = BASE_MODEL_PATH
        else:
            print("⚠️ 基础模型也没找到，尝试在线下载 yolov8n.pt...")
            config['model'] = 'yolov8n.pt'
    
    print(f"🧠 加载模型: {config['model']}")
    model = YOLO(config['model'])

    # 2. 扫描图片
    if not os.path.exists(config['images']):
        print(f"❌ 图片目录不存在: {config['images']}")
        return

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.webp']:
        image_files.extend(glob.glob(os.path.join(config['images'], ext)))

    if not image_files:
        print(f"❌ 未找到图片: {config['images']}")
        return

    print(f"🔍 扫描到 {len(image_files)} 张图片，开始推理...")
    results_list = []

    for img_path in image_files:
        try:
            results = model.predict(img_path, conf=0.25, verbose=False)
        except Exception as e:
            print(f"⚠️ 推理出错 {os.path.basename(img_path)}: {e}")
            continue

        predictions = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label_name = config['labels'].get(cls)
                if not label_name: continue 
                
                # 坐标归一化
                x, y, w, h = box.xywhn[0].tolist()
                
                predictions.append({
                    "from_name": "rect_label", 
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": (x-w/2)*100, "y": (y-h/2)*100, 
                        "width": w*100, "height": h*100, 
                        "rectanglelabels": [label_name]
                    },
                    "score": float(box.conf[0])
                })
        
        # 🔥 生成 Docker 相对路径
        # 物理路径: /data/images/1.jpg
        # 相对路径: images/1.jpg
        # URL: /data/local-files/?d=/data/images/1.jpg
        rel_path = os.path.relpath(img_path, DATA_ROOT)
        ls_url = f"{LS_URL_PREFIX}{rel_path}"

        results_list.append({
            "data": {"image": ls_url},
            "predictions": [{"result": predictions, "score": 0.5}]
        })

    # 3. 保存结果
    os.makedirs(os.path.dirname(config['output']), exist_ok=True)
    with open(config['output'], 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    
    print("-" * 30)
    print(f"✅ 生成完毕: {config['output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    args = parser.parse_args()
    run_inference(args.project)
