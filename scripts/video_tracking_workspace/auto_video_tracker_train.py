import os
import json
import cv2
import requests
import yaml
import argparse
import shutil
from ultralytics import YOLO
from tqdm import tqdm
from urllib.parse import unquote

# === ⚙️ 环境配置 ===
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
LS_URL = os.getenv('LS_URL', 'http://localhost:8080')
LS_API_KEY = os.getenv('LS_API_KEY', '') 
TRAIN_WORKSPACE = os.path.join(DATA_ROOT, "train_tracker_workspace")

# 类别映射 (训练用)
CLASS_MAP = {"Person": 0, "Car": 1, "Defect": 2}

def prepare_dataset(project_id):
    print(f"📡 正在从项目 {project_id} 导出标注数据...")
    headers = {'Authorization': f'Token {LS_API_KEY}'}
    export_url = f"{LS_URL}/api/projects/{project_id}/export?export_type=JSON"
    
    try:
        res = requests.get(export_url, headers=headers)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"❌ 导出失败: {e}。"); return False

    # 清理旧数据
    if os.path.exists(TRAIN_WORKSPACE): shutil.rmtree(TRAIN_WORKSPACE)
    img_dir = os.path.join(TRAIN_WORKSPACE, "images/train")
    lbl_dir = os.path.join(TRAIN_WORKSPACE, "labels/train")
    for d in [img_dir, lbl_dir]: os.makedirs(d, exist_ok=True)

    print("✂️ 正在提取视频关键帧并转换坐标...")
    frame_count = 0
    
    for task in tqdm(data):
        if not task.get('annotations'): continue
        
        # 解析视频路径
        video_url = task['data'].get('video', '')
        decoded_url = unquote(video_url)
        # 兼容 ?d=
        if "?d=" in decoded_url:
            rel_path = decoded_url.split('?d=')[-1]
            # 移除开头的 /data/ 如果有，因为 DATA_ROOT 已经是 /data
            if rel_path.startswith("/data/"): rel_path = rel_path[6:]
            video_path = os.path.join(DATA_ROOT, rel_path)
        else:
            video_path = os.path.join(DATA_ROOT, os.path.basename(decoded_url))
        
        if not os.path.exists(video_path):
            # print(f"🔍 跳过不存在的视频: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        
        # 收集每一帧的标注
        # frame_annotations: { frame_idx: [ "class_id x y w h", ... ] }
        frame_annotations = {}

        for ann in task['annotations']:
            for result in ann['result']:
                # 必须是 videorectangle 且有 labels
                if result['type'] != 'videorectangle': continue
                if 'labels' not in result['value']: continue
                
                label_name = result['value']['labels'][0]
                class_id = CLASS_MAP.get(label_name)
                if class_id is None: continue

                # 遍历 sequence
                for frame_data in result['value']['sequence']:
                    if not frame_data.get('enabled', True): continue
                    
                    f_idx = frame_data['frame'] - 1
                    if f_idx not in frame_annotations: frame_annotations[f_idx] = []
                    
                    x = frame_data['x']
                    y = frame_data['y']
                    w = frame_data['width']
                    h = frame_data['height']
                    
                    # 转 YOLO
                    x_c = (x + w/2) / 100
                    y_c = (y + h/2) / 100
                    w_n = w / 100
                    h_n = h / 100
                    
                    frame_annotations[f_idx].append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        # 提取有标注的帧
        for f_idx, lines in frame_annotations.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            file_id = f"p{project_id}_t{task['id']}_f{f_idx}"
            cv2.imwrite(os.path.join(img_dir, f"{file_id}.jpg"), frame)
            
            with open(os.path.join(lbl_dir, f"{file_id}.txt"), 'w') as f:
                f.write("\n".join(lines))
            frame_count += 1
            
        cap.release()

    # 生成 data.yaml
    yaml_data = {
        'path': TRAIN_WORKSPACE,
        'train': 'images/train',
        'val': 'images/train',
        'nc': 3,
        'names': ['Person', 'Car', 'Defect']
    }
    with open(os.path.join(TRAIN_WORKSPACE, "data.yaml"), 'w') as f:
        yaml.dump(yaml_data, f)
        
    return frame_count > 0

def start_finetuning():
    base_model = "/app/models/yolov8s.pt"
    if not os.path.exists(base_model): base_model = "yolov8n.pt"
    
    print(f"🔥 加载模型: {base_model} ...")
    model = YOLO(base_model)
    
    model.train(
        data=os.path.join(TRAIN_WORKSPACE, "data.yaml"),
        epochs=30, imgsz=640, batch=8,
        project=os.path.join(DATA_ROOT, "runs/tracker_train"),
        name="iter_1", exist_ok=True
    )
    print("✅ 训练完成！新模型在 runs/tracker_train 目录。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=int, required=True)
    args = parser.parse_args()
    
    if prepare_dataset(args.project_id):
        start_finetuning()
    else:
        print("❌ 无有效数据。请在 Label Studio 提交标注 (Submit)。")
