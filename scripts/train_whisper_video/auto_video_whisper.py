import os
import sys
import json
import argparse
from label_studio_sdk.client import LabelStudio

sys.stdout.reconfigure(line_buffering=True)
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)
print(f"📂 P3 (视频语音) 工作目录: {WORK_DIR}")

LS_URL = os.getenv('LS_URL', 'http://localhost:8080')
API_KEY = os.getenv('LS_API_KEY', '')
EXPORT_PATH = "project_export.json" 

def run_pipeline(project_id):
    print(f"🔌 连接 Label Studio (Project {project_id})...")
    try:
        client = LabelStudio(base_url=LS_URL, api_key=API_KEY)
    except Exception as e:
        print(f"❌ 连接失败: {e}"); return

    print(f"🎣 导出数据...")
    try:
        tasks = list(client.projects.exports.as_json(project_id))
        with open(EXPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=4)
        print(f"✅ 导出 {len(tasks)} 条任务")
    except Exception as e:
        print(f"❌ 导出失败: {e}"); return

    python_exe = sys.executable
    print("✂️  步骤1: 准备数据 (prepare_data.py)...")
    
    # 🔥 核心修正：检测返回值，如果失败直接退出
    exit_code = os.system(f"{python_exe} prepare_data.py")
    if exit_code != 0:
        print("🛑 数据准备阶段报错，终止训练。")
        return

    print("🔥 步骤2: 开始微调 (train_whisper.py)...")
    os.system(f"{python_exe} train_whisper.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认 ID 设为 5
    parser.add_argument("--project_id", type=int, default=5)
    args = parser.parse_args()
    run_pipeline(args.project_id)
