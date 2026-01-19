import os
import sys
import json
import shutil
import argparse
from urllib.parse import unquote
import torch

sys.stdout.reconfigure(line_buffering=True)
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)
print(f"📂 P4 工作目录: {WORK_DIR}")

# Docker 变量
LS_URL = os.getenv('LS_URL', 'http://localhost:8080')
API_KEY = os.getenv('LS_API_KEY', '')
DATA_ROOT = os.getenv('DATA_ROOT', '/data')

# P4 图片源
SOURCE_IMG_ROOT = os.path.join(DATA_ROOT, "video_frames")
DATASET_DIR = os.path.abspath("datasets")
YAML_PATH = os.path.abspath("data.yaml")

try:
    from label_studio_sdk.client import LabelStudio
except ImportError:
    print("❌ 未安装 label-studio-sdk"); sys.exit(1)

# 🔥 核心修正：统一类别映射
CLASS_MAP = {"Person": 0, "Car": 1, "Defect": 2}

def convert_ls_to_yolo(ls_result):
    yolo_lines = []
    for region in ls_result:
        if region['type'] != 'rectanglelabels': continue
        value = region['value']
        if not value.get('rectanglelabels'): continue
        
        label_name = value['rectanglelabels'][0]
        if label_name not in CLASS_MAP: continue
        
        class_id = CLASS_MAP[label_name]
        x, y, w, h = value['x'], value['y'], value['width'], value['height']
        
        # 转 YOLO 中心点坐标
        x_center = (x + w / 2) / 100
        y_center = (y + h / 2) / 100
        w_norm = w / 100
        h_norm = h / 100
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    return yolo_lines

def run_pipeline(project_id):
    print(f"🔌 连接 Label Studio: {LS_URL} (Project {project_id})")
    try:
        client = LabelStudio(base_url=LS_URL, api_key=API_KEY)
    except Exception as e:
        print(f"❌ 连接失败: {e}"); return

    print(f"🎣 导出标注数据...")
    try:
        tasks = list(client.projects.exports.as_json(project_id))
    except Exception as e:
        print(f"❌ 导出失败: {e}"); return

    if os.path.exists(DATASET_DIR): shutil.rmtree(DATASET_DIR)
    for d in ["images/train", "labels/train", "images/val", "labels/val"]:
        os.makedirs(os.path.join(DATASET_DIR, d), exist_ok=True)

    print("✂️  转换数据格式...")
    count = 0
    for task in tasks:
        # 获取图片路径
        img_url = task.get('data', {}).get('image', '')
        if not img_url: continue
        
        # 解析路径
        decoded_path = unquote(img_url)
        if "?d=" in decoded_path:
            fname = os.path.basename(decoded_path.split("?d=")[-1])
        else:
            fname = os.path.basename(decoded_path)

        src_path = os.path.join(SOURCE_IMG_ROOT, fname)
        if not os.path.exists(src_path):
            continue

        if not task.get('annotations'): continue
        res = task['annotations'][0].get('result', [])
        
        yolo_data = convert_ls_to_yolo(res)
        if yolo_data:
            txt_name = os.path.splitext(fname)[0] + ".txt"
            for split in ['train', 'val']:
                shutil.copy(src_path, os.path.join(DATASET_DIR, f"images/{split}", fname))
                with open(os.path.join(DATASET_DIR, f"labels/{split}", txt_name), "w") as f:
                    f.write("\n".join(yolo_data))
            count += 1

    print(f"📊 准备了 {count} 个样本")
    if count == 0: 
        print("❌ 无有效样本，请先在 Label Studio 中完成标注并提交 (Submit)。")
        return

    # 生成 YAML
    with open(YAML_PATH, 'w') as f:
        f.write(f"path: {DATASET_DIR}\ntrain: images/train\nval: images/val\nnc: {len(CLASS_MAP)}\nnames:\n")
        for name, idx in CLASS_MAP.items():
            f.write(f"  {idx}: {name}\n")

    # 启动训练
    print("🔥 启动 YOLO 训练...")
    device = 0 if torch.cuda.is_available() else 'cpu'
    from ultralytics import YOLO
    
    # 优先加载离线模型
    local_model = "/app/models/yolov8n.pt"
    if os.path.exists(local_model):
        model = YOLO(local_model)
    else:
        model = YOLO('yolov8n.pt')
        
    model.train(
        data=YAML_PATH, epochs=50, imgsz=640, 
        project=os.path.join(WORK_DIR, 'runs'), name='video_model',
        exist_ok=True, device=device
    )
    print(f"🎉 训练完成！权重保存于: {os.path.join(WORK_DIR, 'runs/video_model/weights/best.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=int, default=4)
    args = parser.parse_args()
    run_pipeline(args.project_id)
