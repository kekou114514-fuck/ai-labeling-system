import os
import sys
import json
import shutil
import argparse
from urllib.parse import unquote

# 强制开启实时日志
sys.stdout.reconfigure(line_buffering=True)

# 锁定工作目录
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)
print(f"📂 P1 工作目录: {WORK_DIR}")

# === Docker 环境变量读取 ===
LS_URL = os.getenv('LS_URL', 'http://localhost:8080')
API_KEY = os.getenv('LS_API_KEY', '') 
DATA_ROOT = os.getenv('DATA_ROOT', '/data')

# P1 图片源路径
SOURCE_IMG_ROOT = os.path.join(DATA_ROOT, "images")
DATASET_DIR = os.path.abspath("datasets")
YAML_PATH = os.path.abspath("data.yaml")
CLASSES_TXT_PATH = os.path.join(DATASET_DIR, "classes.txt")

# 导入 SDK
try:
    from label_studio_sdk.client import LabelStudio
except ImportError:
    print("❌ 未安装 label-studio-sdk")
    sys.exit(1)

# 🔥 核心标签映射 (根据您的 XML 配置)
# 策略：将不同的标注形式映射到 YOLO 的类别 ID
CLASS_MAP = {
    # 矩形框
    "Product_A": 0, 
    "Product_B": 1, 
    "Defect": 2, 
    "person": 3,
    
    # 多边形 (Polygon) -> 映射到对应类别
    "Defect_Shape": 2,  # 所有的缺陷形状都归为 "Defect" 类
    "Component": 4,     # 新增类别 Component
    
    # 关键点 (KeyPoint) -> 映射为新类别
    "Corner_Point": 5,
    
    # 笔刷 (Brush) -> 映射到缺陷 (注意：笔刷处理较复杂，暂作简单映射)
    "Surface_Area": 2
}

# 反向映射用于生成 names
ID_TO_NAME = {v: k for k, v in CLASS_MAP.items()} 
# 修正反向映射，优先保留主名称
ID_TO_NAME[2] = "Defect"

def xywh_to_yolo(x, y, w, h):
    """将 LabelStudio 的百分比坐标转换为 YOLO 归一化中心点坐标"""
    x_center = (x + w / 2) / 100.0
    y_center = (y + h / 2) / 100.0
    w_norm = w / 100.0
    h_norm = h / 100.0
    return x_center, y_center, w_norm, h_norm

def convert_ls_to_yolo(ls_result):
    yolo_lines = []
    
    for region in ls_result:
        r_type = region.get('type')
        value = region.get('value', {})
        
        # 1. 获取标签名称
        labels = value.get('rectanglelabels') or \
                 value.get('polygonlabels') or \
                 value.get('brushlabels') or \
                 value.get('keypointlabels')
                 
        if not labels: continue
        label_name = labels[0]
        
        if label_name not in CLASS_MAP: 
            continue # 忽略未定义标签
            
        class_id = CLASS_MAP[label_name]
        
        # 2. 处理矩形 (Rectangle)
        if r_type == 'rectanglelabels':
            x, y, w, h = value['x'], value['y'], value['width'], value['height']
            xc, yc, wn, hn = xywh_to_yolo(x, y, w, h)
            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            
        # 3. 处理多边形 (Polygon) -> 转为外接矩形 bbox
        elif r_type == 'polygonlabels':
            points = value.get('points', [])
            if not points: continue
            
            # 提取所有点的 x 和 y
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            w = max_x - min_x
            h = max_y - min_y
            
            xc, yc, wn, hn = xywh_to_yolo(min_x, min_y, w, h)
            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        # 4. 处理关键点 (KeyPoint) -> 转为微小矩形框 (1% 大小)
        elif r_type == 'keypointlabels':
            x, y = value['x'], value['y']
            # 创建一个 1% x 1% 的小框
            w, h = 1.0, 1.0 
            # 居中调整
            start_x = x - (w/2)
            start_y = y - (h/2)
            xc, yc, wn, hn = xywh_to_yolo(start_x, start_y, w, h)
            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    return yolo_lines

def run_pipeline(project_id):
    print(f"🔌 连接 Label Studio: {LS_URL}")
    try:
        client = LabelStudio(base_url=LS_URL, api_key=API_KEY)
    except Exception as e:
        print(f"❌ 连接失败: {e}"); return

    # 清理并重建数据集目录
    if os.path.exists(DATASET_DIR): shutil.rmtree(DATASET_DIR)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_DIR, f"images/{split}"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, f"labels/{split}"), exist_ok=True)

    print(f"🎣 导出项目 {project_id}...")
    try:
        tasks = client.projects.exports.as_json(project_id)
    except Exception as e:
        print(f"❌ 导出失败: {e}"); return
        
    count = 0

    for task in tasks:
        img_url = task['data'].get('image', '')
        if not img_url: continue
        
        decoded_path = unquote(img_url)
        fname = os.path.basename(decoded_path)
        
        # 查找图片文件
        found = False
        src_path = os.path.join(SOURCE_IMG_ROOT, fname)
        
        # 支持 ?d= 格式的路径处理
        if "?d=" in decoded_path:
            real_path = decoded_path.split("?d=")[-1]
            if os.path.exists(real_path):
                src_path = real_path
                found = True
            elif os.path.exists(os.path.join(DATA_ROOT, real_path.lstrip('/'))):
                src_path = os.path.join(DATA_ROOT, real_path.lstrip('/'))
                found = True

        if not found and os.path.exists(src_path):
            found = True

        if not found:
            # 递归搜索
            for root, dirs, files in os.walk(SOURCE_IMG_ROOT):
                if fname in files:
                    src_path = os.path.join(root, fname); found = True; break
        
        if not found: 
            # print(f"⚠️ 图片未找到: {fname}")
            continue

        if not task.get('annotations'): continue
        res = task['annotations'][0].get('result', [])
        if not res: continue

        # 🔥 转换多种标注类型为 YOLO 格式
        yolo_data = convert_ls_to_yolo(res)
        
        if yolo_data:
            txt_name = os.path.splitext(fname)[0] + ".txt"
            for split in ['train', 'val']:
                shutil.copy(src_path, os.path.join(DATASET_DIR, f"images/{split}", fname))
                with open(os.path.join(DATASET_DIR, f"labels/{split}", txt_name), "w") as f:
                    f.write("\n".join(yolo_data))
            count += 1

    print(f"📊 准备了 {count} 个样本 (支持多边形/关键点转化)")
    
    # 生成 classes.txt
    unique_ids = sorted(list(set(CLASS_MAP.values())))
    with open(CLASSES_TXT_PATH, 'w') as f:
        for idx in unique_ids:
            name = ID_TO_NAME.get(idx, f"class_{idx}")
            f.write(f"{name}\n")

    # 生成 data.yaml
    with open(YAML_PATH, 'w') as f:
        f.write(f"path: {DATASET_DIR}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(unique_ids)}\nnames:\n")
        for idx in unique_ids:
            name = ID_TO_NAME.get(idx, f"class_{idx}")
            f.write(f"  {idx}: {name}\n")

    if count == 0:
        print("❌ 无有效样本，终止训练。"); return

    print("🔥 启动 YOLO 训练 (train.py)...")
    os.system(f"{sys.executable} train.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=int, required=True)
    args = parser.parse_args()
    run_pipeline(args.project_id)
