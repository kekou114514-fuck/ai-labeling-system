import json
import os
import sys
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
import urllib.parse

# === ⚙️ 路径配置 ===
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
EXPORT_FILE = "project_export.json" 
OUTPUT_DIR = "dataset"
# ===================

def prepare_dataset():
    # 初始化目录
    os.makedirs(os.path.join(OUTPUT_DIR, "audio"), exist_ok=True)

    if not os.path.exists(EXPORT_FILE):
        print(f"❌ 找不到导出文件 {EXPORT_FILE}")
        return

    with open(EXPORT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = []
    print(f"✂️  开始处理 {len(data)} 个任务...")

    for task in tqdm(data):
        # 1. 获取音频路径
        audio_url = task.get('data', {}).get('audio', '') or task.get('data', {}).get('audio_url', '')
        if not audio_url: continue

        decoded_url = urllib.parse.unquote(audio_url)
        audio_path = ""

        # 🔥【终极修复】暴力路径匹配逻辑
        # 无论 URL 长什么样，我们要找的文件一定在 /data/audio/ 下面
        
        # 1. 提取纯文件名 (例如: 曾侯乙clock1号.mp3)
        if "?d=" in decoded_url:
            raw_path_segment = decoded_url.split("?d=")[-1] # 可能是 data/audio/xxx.mp3
            filename = os.path.basename(raw_path_segment)
        else:
            filename = os.path.basename(decoded_url)

        # 2. 构造标准绝对路径 /data/audio/filename
        # 即使 Label Studio 传回的是 data/audio/xxx，我们也强制指向 /data/audio/xxx
        candidate_path = os.path.join(DATA_ROOT, "audio", filename)

        # 3. 验证存在性
        if os.path.exists(candidate_path):
            audio_path = candidate_path
        else:
            # 备选方案：万一文件不在 audio 文件夹里，而在根目录？
            candidate_path_root = os.path.join(DATA_ROOT, filename)
            if os.path.exists(candidate_path_root):
                audio_path = candidate_path_root

        # 最终检查
        if not audio_path:
            print(f"⚠️ 文件未找到: {filename} (尝试路径: {candidate_path})")
            continue

        try:
            # 加载原始音频
            y, sr = librosa.load(audio_path, sr=16000)
            
            # 2. 遍历标注结果
            found_annotation = False
            for annotation in task.get('annotations', []):
                for result in annotation.get('result', []):
                    # 必须是 textarea (文本转写)
                    if result.get('type') != 'textarea':
                        continue
                    
                    text_val = result.get('value', {}).get('text', [])
                    text = text_val[0] if text_val else ""
                    if not text or "正在转写" in text: continue

                    # 获取时间戳
                    start = result.get('value', {}).get('start', 0)
                    end = result.get('value', {}).get('end', len(y)/sr)
                    
                    # 3. 切片音频
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    y_chunk = y[start_sample:end_sample]

                    # 忽略太短 (<0.1s)
                    if len(y_chunk) < 1600: continue

                    chunk_name = f"t{task['id']}_{result['id']}.wav"
                    save_path = os.path.join(OUTPUT_DIR, "audio", chunk_name)
                    sf.write(save_path, y_chunk, sr)

                    metadata.append({
                        "file_name": f"audio/{chunk_name}", # CSV里存相对路径
                        "sentence": text
                    })
                    found_annotation = True
            
            if not found_annotation:
                pass 

        except Exception as e:
            print(f"⚠️ 处理出错: {e}")

    # 4. 保存元数据
    if metadata:
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
        print(f"✅ 成功提取 {len(metadata)} 个音频片段！")
    else:
        print("❌ 未提取到有效数据。请检查：")
        print("1. Label Studio 里是否确实点了 Submit")
        print("2. 标注的文本框里是否有内容")
        sys.exit(1)

if __name__ == "__main__":
    prepare_dataset()
