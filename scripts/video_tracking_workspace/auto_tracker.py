import os
import cv2
import json
import uuid
import sys
import random
import glob
from collections import defaultdict
from ultralytics import YOLO

# === Docker 适配配置 ===
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
# 视频存放目录
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
# 结果输出目录
OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LS_URL_PREFIX = "/data/local-files/?d=/data/"

# XML 定义
XML_BOX_NAME = "box"
XML_LABEL_NAME = "labels"
LABEL_MOVING = "Object_Moving"
LABEL_STATIC = "Object_Static"
MOVEMENT_SENSITIVITY = 0.5 

def run_tracking(video_path, output_json):
    # 优先加载离线模型
    local_seg = "/app/models/yolov8n-seg.pt"
    local_det = "/app/models/yolov8n.pt"
    
    if os.path.exists(local_seg):
        print(f"🧠 加载分割模型: {local_seg}")
        model = YOLO(local_seg)
    elif os.path.exists(local_det):
        print(f"⚠️ 未找到seg模型，使用检测模型: {local_det}")
        model = YOLO(local_det)
    else:
        print("⚠️ 未找到本地模型，下载 yolov8n.pt")
        model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"❌ 无法打开视频: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"🔥 开始追踪: {os.path.basename(video_path)}")
    results = model.track(source=video_path, persist=True, stream=True, verbose=False)
    tracks_data = defaultdict(list) 

    # 数据采集
    for frame_idx, r in enumerate(results):
        if not r.boxes or r.boxes.id is None: continue
        boxes = r.boxes.xywh.cpu().numpy()
        track_ids = r.boxes.id.int().cpu().tolist()
        img_h, img_w = r.orig_shape[0], r.orig_shape[1]

        for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
            x_center, y_center, width, height = box
            # 归一化
            x = (x_center - width / 2) / img_w * 100
            y = (y_center - height / 2) / img_h * 100
            w = width / img_w * 100
            h = height / img_h * 100
            
            tracks_data[track_id].append({
                "frame": frame_idx + 1,
                "enabled": True,
                "rotation": 0,
                "x": float(x), "y": float(y), "width": float(w), "height": float(h),
                "time": float(frame_idx / fps) if fps > 0 else 0.0
            })

    # 生成标注
    ls_results = []
    for track_id, sequence_data in tracks_data.items():
        if not sequence_data: continue
        
        # 行为判定
        first = sequence_data[0]
        last = sequence_data[-1]
        dist = ((last['x'] - first['x'])**2 + (last['y'] - first['y'])**2)**0.5
        span = last['frame'] - first['frame']
        
        final_label = LABEL_STATIC
        if span > 0:
            speed = dist / span
            if speed > (MOVEMENT_SENSITIVITY / 10.0):
                final_label = LABEL_MOVING

        shared_id = str(uuid.uuid4())[:8]
        # 轨迹
        ls_results.append({
            "id": shared_id, "from_name": XML_BOX_NAME, "to_name": "video", "type": "videorectangle",
            "value": {"sequence": sequence_data, "original_width": orig_w, "original_height": orig_h}
        })
        # 标签
        ls_results.append({
            "id": shared_id, "from_name": XML_LABEL_NAME, "to_name": "video", "type": "labels",
            "value": {"labels": [final_label], "sequence": sequence_data, "original_width": orig_w, "original_height": orig_h}
        })

    # 封装
    rel_path = os.path.relpath(video_path, DATA_ROOT)
    final_output = [{
        "data": { "video": f"{LS_URL_PREFIX}{rel_path}" },
        "annotations": [{"result": ls_results, "ground_truth": False}]
    }]

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"✅ 生成: {output_json}")

if __name__ == "__main__":
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ 视频目录不存在: {VIDEO_DIR}")
        sys.exit(1)
        
    files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) + glob.glob(os.path.join(VIDEO_DIR, "*.avi"))
    
    if not files:
        print(f"❌ 未找到视频文件")
    else:
        for v_path in files:
            fname = os.path.basename(v_path)
            out_path = os.path.join(OUTPUT_DIR, f"track_{fname}.json")
            run_tracking(v_path, out_path)
