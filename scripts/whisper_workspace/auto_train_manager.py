import os
import sys
import json
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
    print(f"🔌 连接 Label Studio (Project {project_id})...")
    try:
        client = LabelStudio(base_url=LS_URL, api_key=API_KEY)
    except Exception as e:
        print(f"❌ 连接失败: {e}"); return

    print(f"🎣 导出标注数据...")
    try:
        # 导出为 JSON 格式
        tasks = client.projects.exports.as_json(project_id)
        final_data = list(tasks)
        if not final_data:
            print("❌ 导出数据为空，请先在 Label Studio 中完成标注并提交 (Submit)。")
            return
        
        with open(EXPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 成功导出 {len(final_data)} 个任务")
    except Exception as e:
        print(f"❌ 导出异常: {e}"); return

    # 调用数据准备
    python_exe = sys.executable
    print("✂️  步骤1: 音频切片与清洗 (prepare_data.py)...")
    if os.system(f"{python_exe} prepare_data.py") != 0:
        print("❌ 数据准备失败，终止训练。"); return

    # 调用训练
    print("🔥 步骤2: 启动 Whisper 微调 (train_whisper.py)...")
    os.system(f"{python_exe} train_whisper.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认 ID 设为 3
    parser.add_argument("--project_id", type=int, default=3)
    args = parser.parse_args()
    run_auto_pipeline(args.project_id)
