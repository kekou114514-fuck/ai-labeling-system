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
    sys.exit(1)

CLASS_MAP = {"defect": 0, "scratch": 1}

def convert_ls_to_yolo(ls_result, img_width, img_height):
    yolo_lines = []
    for region in ls_result:
        if region['type'] != 'rectanglelabels': continue
        value = region['value']
        if not value.get('rectanglelabels'): continue
        label_name = value['rectanglelabels'][0]
        if label_name not in CLASS_MAP: continue
        class_id = CLASS_MAP[label_name]
        x, y, w, h = value['x'], value['y'], value['width'], value['height']
        x_center = (x + w / 2) / 100
        y_center = (y + h / 2) / 100
        w_norm = w / 100
        h_norm = h / 100
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    return yolo_lines

def run_pipeline(project_id):
    print(f"🔌 连接 Label Studio: {LS_URL}")
    try:
        client = LabelStudio(base_url=LS_URL, api_key=API_KEY)
        client.users.whoami()
    except Exception as e:
        print(f"❌ 连接失败: {e}"); return

    print(f"🎣 导出项目 {project_id}...")
    try:
        tasks = list(client.projects.exports.as_json(project_id))
        print(f"✅ {len(tasks)} 条任务")
    except Exception as e:
        print(f"❌ 导出失败: {e}"); return

    if os.path.exists(DATASET_DIR): shutil.rmtree(DATASET_DIR)
    for d in ["images/train", "labels/train", "images/val", "labels/val"]:
        os.makedirs(os.path.join(DATASET_DIR, d), exist_ok=True)

    print("✂️  转换数据...")
    count = 0
    for task in tasks:
        img_url = task.get('data', {}).get('image', '')
        if not img_url: continue
        fname = os.path.basename(unquote(img_url).split('?')[0])

        src_path = os.path.join(SOURCE_IMG_ROOT, fname)
        if not os.path.exists(src_path):
            continue # 如果没找到图片就跳过

        if not task.get('annotations'): continue
        res = task['annotations'][0].get('result', [])
        
        orig_w = res[0].get('original_width', 1920) if res else 1920
        orig_h = res[0].get('original_height', 1080) if res else 1080
        
        yolo_data = convert_ls_to_yolo(res, orig_w, orig_h)
        if yolo_data:
            txt_name = os.path.splitext(fname)[0] + ".txt"
            for split in ['train', 'val']:
                shutil.copy(src_path, os.path.join(DATASET_DIR, f"images/{split}", fname))
                with open(os.path.join(DATASET_DIR, f"labels/{split}", txt_name), "w") as f:
                    f.write("\n".join(yolo_data))
            count += 1

    print(f"📊 样本数: {count}")
    if count == 0: return

    # 生成 YAML
    with open(YAML_PATH, 'w') as f:
        f.write(f"path: {DATASET_DIR}\ntrain: images/train\nval: images/val\nnc: {len(CLASS_MAP)}\nnames:\n")
        for name, idx in CLASS_MAP.items():
            f.write(f"  {idx}: {name}\n")

    # 内置训练逻辑
    print("🔥 启动 YOLO 训练...")
    device = 0 if torch.cuda.is_available() else 'cpu'
    from ultralytics import YOLO
    
    # 优先加载离线模型
    local_model = "/app/models/yolov8n.pt"
    model_path = local_model if os.path.exists(local_model) else "yolov8n.pt"
    
    model = YOLO(model_path)
    model.train(
        data=YAML_PATH, epochs=50, imgsz=640, project='.', name='run_video_v1',
        exist_ok=True, device=device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=int, default=4)
    args = parser.parse_args()
    run_pipeline(args.project_id)
