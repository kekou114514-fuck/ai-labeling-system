import os
import cv2
import json
import uuid
import sys
import glob
from collections import defaultdict
from ultralytics import YOLO

# === ⚙️ 路径配置 ===
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Label Studio 前缀
LS_URL_PREFIX = "/data/local-files/?d=data/"

# 核心参数
TRACK_CONF = 0.1      
TRACK_IOU = 0.5       
IMG_SIZE = 1280       

# 类别映射
CLASS_MAP = {
    0: "Person",
    1: "bicycle",
    2: "Car", 3: "Car", 5: "Car", 7: "Car"
}

def run_tracking_logic(video_path):
    # 1. 🔥 模型加载逻辑 (优先查找 models 目录)
    # 在 Docker 容器中，宿主机的 models 目录通常映射为 /app/models
    model_candidates = [
        "/app/models/yolov8m.pt",       # 优先：统一模型库
        "/app/models/yolov8n.pt",       # 备选：Nano 模型
        "yolov8m.pt"                    # 再次：当前目录或在线下载
    ]
    
    model_path = "yolov8m.pt" # 默认值
    for p in model_candidates:
        if os.path.exists(p):
            model_path = p
            break
            
    print(f"🧠 加载模型: {model_path}")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"⚠️ 加载失败: {e}，尝试在线下载 yolov8m.pt...")
        model = YOLO("yolov8m.pt")
    
    # 2. 读取视频信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"🎬 正在深度追踪: {os.path.basename(video_path)} ...")
    
    # 3. 运行 ByteTrack
    results = model.track(
        source=video_path, 
        persist=True, 
        conf=TRACK_CONF, 
        iou=TRACK_IOU,
        imgsz=IMG_SIZE,
        tracker="bytetrack.yaml", 
        verbose=False,
        stream=True
    )
    
    # 4. 整理轨迹数据
    tracks_data = defaultdict(list)
    track_id_to_label = {}

    for frame_idx, r in enumerate(results):
        if not r.boxes: continue
        
        boxes = r.boxes.xywh.cpu().numpy()
        
        if r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1] * len(boxes)

        cls_ids = r.boxes.cls.int().cpu().tolist()

        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
            label_name = CLASS_MAP.get(cls_id)
            if not label_name: continue 
            
            if track_id != -1:
                track_id_to_label[track_id] = label_name

            x_c, y_c, w, h = box
            
            frame_data = {
                "frame": frame_idx + 1,
                "enabled": True, 
                "rotation": 0,
                "x": float((x_c - w / 2) / orig_w * 100),
                "y": float((y_c - h / 2) / orig_h * 100),
                "width": float(w / orig_w * 100),
                "height": float(h / orig_h * 100),
                "time": float(frame_idx / fps) if fps > 0 else 0.0
            }
            
            if track_id != -1:
                tracks_data[track_id].append(frame_data)

    # 5. 生成 JSON
    ls_results = []
    print(f"📊 捕捉到 {len(tracks_data)} 条轨迹")

    for track_id, sequence in tracks_data.items():
        label = track_id_to_label.get(track_id, "Defect")
        shared_id = str(uuid.uuid4())[:8]
        
        ls_results.append({
            "id": shared_id, 
            "from_name": "box", "to_name": "video", "type": "videorectangle",
            "value": {
                "sequence": sequence, 
                "labels": [label],
                "original_width": orig_w, "original_height": orig_h
            }
        })
        ls_results.append({
            "id": shared_id, 
            "from_name": "label", "to_name": "video", "type": "labels",
            "value": {
                "sequence": sequence,
                "labels": [label], 
                "original_width": orig_w, "original_height": orig_h
            }
        })

    rel_path = os.path.relpath(video_path, DATA_ROOT)
    if rel_path.startswith("/"): rel_path = rel_path[1:]
    
    return {
        "data": { "video": f"{LS_URL_PREFIX}{rel_path}" },
        "predictions": [{
            "model_version": "yolo_tracker_v2_aggressive", 
            "score": 0.5,
            "result": ls_results
        }]
    }

if __name__ == "__main__":
    extensions = ['*.mp4', '*.avi', '*.mov']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    
    if not video_files:
        print(f"❌ 未找到视频文件: {VIDEO_DIR}")
        sys.exit(0)

    all_tasks = []
    for v in video_files:
        res = run_tracking_logic(v)
        if res: all_tasks.append(res)
    
    if all_tasks:
        out_file = os.path.join(OUTPUT_DIR, "pre_annotations_tracking.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(all_tasks, f, indent=2, ensure_ascii=False)
        print(f"🚀 追踪完成！JSON 已保存: {out_file}")
    else:
        print("⚠️ 未生成任何结果。")
