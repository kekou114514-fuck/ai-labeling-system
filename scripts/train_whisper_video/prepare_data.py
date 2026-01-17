import json
import os
import sys
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
import urllib.parse 

sys.stdout.reconfigure(line_buffering=True)

# ==========================================
# ⚙️ Docker 路径配置
# ==========================================
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
EXPORT_FILE = "./project_export.json" 
# 指向 project_data/video_audio
AUDIO_DIR = os.path.join(DATA_ROOT, "video_audio")  
OUTPUT_DIR = "./dataset"
# ==========================================

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
    print(f"✂️  开始处理 P3 任务 (音频源: {AUDIO_DIR})...")

    for task in tqdm(data):
        audio_url = task.get('data', {}).get('audio', '')
        if not audio_url: continue

        decoded_url = urllib.parse.unquote(audio_url)
        fname = os.path.basename(decoded_url).split('?')[0]
        audio_path = os.path.join(AUDIO_DIR, fname)
        
        if not os.path.exists(audio_path):
            found = False
            for root, _, files in os.walk(AUDIO_DIR):
                if fname in files:
                    audio_path = os.path.join(root, fname); found = True; break
            if not found: continue

        try:
            audio = AudioSegment.from_file(audio_path)
            for ann in task.get('annotations', []):
                for res in ann.get('result', []):
                    if res.get('type') == 'textarea':
                        text = res.get('value', {}).get('text', [''])[0].strip()
                        if not text or "正在转写" in text: continue
                        
                        # 智能判断时间
                        if 'start' in res['value'] and 'end' in res['value']:
                            start_ms = res['value']['start'] * 1000
                            end_ms = res['value']['end'] * 1000
                        else:
                            start_ms = 0
                            end_ms = len(audio)
                        
                        chunk_name = f"task{task['id']}_{res['id']}.wav"
                        save_path = os.path.join(OUTPUT_DIR, "audio", chunk_name)
                        
                        audio[start_ms:end_ms].export(save_path, format="wav")
                        metadata.append({"file_name": f"audio/{chunk_name}", "sentence": text})

        except Exception as e:
            print(f"⚠️ 错误: {e}")

    if metadata:
        pd.DataFrame(metadata).to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
        print(f"✅ 成功生成 {len(metadata)} 条训练数据！")
    else:
        print("❌ 未提取到数据。")

if __name__ == "__main__":
    prepare_dataset()
