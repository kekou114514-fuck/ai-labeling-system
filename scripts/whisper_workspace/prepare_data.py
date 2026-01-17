import json
import os
import sys
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
import urllib.parse

sys.stdout.reconfigure(line_buffering=True)

# === ⚙️ Docker 路径配置 ===
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
EXPORT_FILE = "./project_export.json" 
# 指向 project_data/audio
AUDIO_DIR = os.path.join(DATA_ROOT, "audio") 
OUTPUT_DIR = "./dataset"
# ===================

def prepare_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        os.makedirs(os.path.join(OUTPUT_DIR, "audio"))

    if not os.path.exists(EXPORT_FILE):
        print(f"❌ 错误：找不到 {EXPORT_FILE}")
        return

    with open(EXPORT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = []
    print(f"✂️  开始处理 {len(data)} 个任务 (音频源: {AUDIO_DIR})...")

    for task in tqdm(data):
        audio_url = task.get('data', {}).get('audio', '')
        if not audio_url: continue

        decoded_url = urllib.parse.unquote(audio_url)
        fname = os.path.basename(decoded_url).split('?')[0]

        # 在 /data/audio 查找文件
        audio_path = os.path.join(AUDIO_DIR, fname)
        
        if not os.path.exists(audio_path):
            # 尝试递归搜索
            found = False
            for root, _, files in os.walk(AUDIO_DIR):
                if fname in files:
                    audio_path = os.path.join(root, fname)
                    found = True; break
            if not found: continue

        try:
            audio = AudioSegment.from_file(audio_path)
            for ann in task.get('annotations', []):
                for res in ann.get('result', []):
                    if res.get('type') == 'textarea':
                        text = res.get('value', {}).get('text', [''])[0].strip()
                        if not text or "正在转写" in text: continue
                            
                        start_ms = res['value'].get('start', 0) * 1000
                        end_ms = res['value'].get('end', len(audio)/1000) * 1000
                        
                        chunk_name = f"task{task['id']}_{res['id']}.wav"
                        save_path = os.path.join(OUTPUT_DIR, "audio", chunk_name)
                        
                        audio[start_ms:end_ms].export(save_path, format="wav")
                        metadata.append({"file_name": f"audio/{chunk_name}", "sentence": text})
        except Exception as e:
            print(f"⚠️ 错误: {e}")

    if metadata:
        pd.DataFrame(metadata).to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
        print(f"✅ 成功切分 {len(metadata)} 条数据！")
    else:
        print("❌ 未提取到数据。请检查音频文件是否已放入 project_data/audio")

if __name__ == "__main__":
    prepare_dataset()
