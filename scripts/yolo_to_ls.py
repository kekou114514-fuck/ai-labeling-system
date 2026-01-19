import os
import glob
import json
import argparse
from ultralytics import YOLO

# ==========================================
# ⚙️ Docker 适配配置
# ==========================================
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
LS_URL_PREFIX = "/data/local-files/?d=/data/"
BASE_MODEL_PATH = "/app/models/yolov8n.pt"

def run_inference(project_type):
    # === P1: 产品图片 ===
    if project_type in ['1', '2']:
        print(f"📦 模式: 项目 {project_type} (产品图片检测)")
        config = {
            "images": os.path.join(DATA_ROOT, "images"),
            "model": os.path.join(DATA_ROOT, "scripts/yolo_workspace/runs/detect/my_defect_project/weights/best.pt"),
            "output": os.path.join(DATA_ROOT, "outputs/pre_annotations_images.json"),
            "labels": {0: "Product_A", 1: "Product_B", 2: "Defect", 3: "person", 4: "Component", 5: "Corner_Point"}
        }
        
    # === P4: 视频抽帧 (ID 4) ===
    elif project_type == '4':
        print("🎬 模式: 项目 4 (视频抽帧图片)")
        config = {
            "images": os.path.join(DATA_ROOT, "video_frames"),
            # 优先用训练好的模型
            "model": os.path.join(DATA_ROOT, "scripts/train_yolo_video/runs/video_model/weights/best.pt"),
            "output": os.path.join(DATA_ROOT, "outputs/pre_annotations_video_frames.json"),
            # 🔥 必须与 auto_video_yolo.py 一致
            "labels": {0: "Person", 1: "Car", 2: "Defect"} 
        }
    else:
        print(f"❌ 未知项目类型: {project_type}"); return

    # 1. 模型加载逻辑
    used_base_model = False
    if not os.path.exists(config['model']):
        print(f"⚠️ 专属模型未找到: {config['model']}")
        if os.path.exists(BASE_MODEL_PATH):
            print(f"🔄 切换为基础模型: {BASE_MODEL_PATH}")
            config['model'] = BASE_MODEL_PATH
        else:
            config['model'] = 'yolov8n.pt'
        used_base_model = True
    
    print(f"🧠 加载模型: {config['model']}")
    try:
        model = YOLO(config['model'])
    except Exception as e:
        print(f"❌ 模型加载失败: {e}"); return

    # 2. 扫描图片
    if not os.path.exists(config['images']):
        print(f"❌ 目录不存在: {config['images']}"); return

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_files.extend(glob.glob(os.path.join(config['images'], ext)))
        image_files.extend(glob.glob(os.path.join(config['images'], ext.upper())))

    if not image_files:
        print(f"❌ 未找到图片: {config['images']}"); return

    print(f"🔍 扫描到 {len(image_files)} 张图片，开始推理...")
    results_list = []

    for img_path in image_files:
        try:
            results = model.predict(img_path, conf=0.25, verbose=False)
        except Exception as e:
            print(f"⚠️ 跳过 {os.path.basename(img_path)}: {e}"); continue

        predictions = []
        for result in results:
            for box in result.boxes:
                try:
                    cls_id = int(box.cls[0])
                    label_name = config['labels'].get(cls_id)
                    
                    # 🔥 智能兜底：如果用的是基础 yolov8n，自动映射 COCO 类别到我们的 XML
                    if used_base_model or "yolov8n" in str(config['model']):
                        if cls_id == 0: label_name = "Person"  # COCO 0 -> Person
                        elif cls_id == 2: label_name = "Car"   # COCO 2 -> Car
                        elif project_type == '1' and cls_id == 5: label_name = "Product_A" # 示例: 把巴士当产品A (仅演示)
                        # 其他无关物体过滤掉
                    
                    if not label_name: continue

                    x, y, w, h = box.xywhn[0].tolist()
                    predictions.append({
                        "from_name": "label", "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": (x-w/2)*100, "y": (y-h/2)*100, 
                            "width": w*100, "height": h*100, 
                            "rectanglelabels": [label_name]
                        },
                        "score": float(box.conf[0])
                    })
                except: continue
        
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
    
    print(f"✅ 生成完毕: {config['output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    args = parser.parse_args()
    run_inference(args.project)
