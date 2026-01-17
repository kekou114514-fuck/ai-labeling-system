import os
import sys
import json
import time
import argparse
from label_studio_sdk.client import LabelStudio

sys.stdout.reconfigure(line_buffering=True)
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)
print(f"📂 P2 工作目录: {WORK_DIR}")

LS_URL = os.getenv('LS_URL', 'http://localhost:8080')
API_KEY = os.getenv('LS_API_KEY', '')
EXPORT_PATH = "project_export.json" 

def run_auto_pipeline(project_id):
    print(f"🔌 连接 Label Studio: {LS_URL}")
    try:
        client = LabelStudio(base_url=LS_URL, api_key=API_KEY)
        client.users.whoami()
    except Exception as e:
        print(f"❌ 连接失败: {e}"); return

    print(f"🎣 导出项目 {project_id}...")
    try:
        tasks = client.projects.exports.as_json(project_id)
        final_data = list(tasks)
        with open(EXPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 导出 {len(final_data)} 条数据")
    except Exception as e:
        print(f"❌ 导出失败: {e}"); return

    # 调用步骤 3.2 已经准备好的 prepare_data.py
    python_exe = sys.executable
    print("✂️  调用数据准备 (prepare_data.py)...")
    if os.system(f"{python_exe} prepare_data.py") != 0:
        print("❌ 数据准备失败"); return

    print("🔥 调用微调 (train_whisper.py)...")
    os.system(f"{python_exe} train_whisper.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=int, default=2)
    args = parser.parse_args()
    run_auto_pipeline(args.project_id)
