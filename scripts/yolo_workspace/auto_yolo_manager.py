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
API_KEY = os.getenv('LS_API_KEY', '') # 必须在 env 文件中配置
DATA_ROOT = os.getenv('DATA_ROOT', '/data')

# P1 图片源路径
SOURCE_IMG_ROOT = os.path.join(DATA_ROOT, "images")
DATASET_DIR = os.path.abspath("datasets")
YAML_PATH = os.path.abspath("data.yaml")

# 导入 SDK
try:
    from label_studio_sdk.client import LabelStudio
except ImportError:
    print("❌ 未安装 label-studio-sdk")
    sys.exit(1)

CLASS_MAP = {"物体框(Box)": 0, "文字区域": 1, "复杂轮廓(Poly)": 2}

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
        print(f"❌ 连接失败: {e}\n👉 请检查 env 文件中的 API KEY 是否正确。"); return

    print(f"🎣 导出项目 {project_id} 数据...")
    try:
        tasks = client.projects.exports.as_json(project_id)
        # 兼容列表或生成器
        tasks = list(tasks)
        print(f"✅ 获取到 {len(tasks)} 条任务")
    except Exception as e:
        print(f"❌ 导出失败: {e}"); return

    # 清理数据集目录
    if os.path.exists(DATASET_DIR): shutil.rmtree(DATASET_DIR)
    for d in ["images/train", "labels/train", "images/val", "labels/val"]:
        os.makedirs(os.path.join(DATASET_DIR, d), exist_ok=True)

    print("✂️  开始转换...")
    count = 0
    for task in tasks:
        # 获取文件名: /data/local-files/?d=/data/images/1.jpg -> 1.jpg
        img_url = task.get('data', {}).get('image', '')
        if not img_url: continue
        fname = os.path.basename(unquote(img_url).split('?')[0])

        src_path = os.path.join(SOURCE_IMG_ROOT, fname)
        if not os.path.exists(src_path):
            # 尝试在子目录查找
            found = False
            for root, _, files in os.walk(SOURCE_IMG_ROOT):
                if fname in files:
                    src_path = os.path.join(root, fname); found = True; break
            if not found: continue

        if not task.get('annotations'): continue
        res = task['annotations'][0].get('result', [])
        if not res: continue

        # 转换坐标
        orig_w = res[0].get('original_width', 1920)
        orig_h = res[0].get('original_height', 1080)
        yolo_data = convert_ls_to_yolo(res, orig_w, orig_h)
        
        if yolo_data:
            txt_name = os.path.splitext(fname)[0] + ".txt"
            for split in ['train', 'val']:
                shutil.copy(src_path, os.path.join(DATASET_DIR, f"images/{split}", fname))
                with open(os.path.join(DATASET_DIR, f"labels/{split}", txt_name), "w") as f:
                    f.write("\n".join(yolo_data))
            count += 1

    print(f"📊 准备了 {count} 个样本")
    if count == 0:
        print("❌ 无有效样本，终止训练。"); return

    # 生成 YAML
    with open(YAML_PATH, 'w') as f:
        f.write(f"path: {DATASET_DIR}\ntrain: images/train\nval: images/val\nnc: {len(CLASS_MAP)}\nnames:\n")
        for name, idx in CLASS_MAP.items():
            f.write(f"  {idx}: {name}\n")

    print("🔥 调用 train.py 开始训练...")
    os.system(f"{sys.executable} train.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=int, default=1)
    args = parser.parse_args()
    run_pipeline(args.project_id)
