import json
import os
import sys
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
import urllib.parse 

sys.stdout.reconfigure(line_buffering=True)

# === 配置 ===
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
EXPORT_FILE = "./project_export.json" 
AUDIO_DIR = os.path.join(DATA_ROOT, "video_audio")  
OUTPUT_DIR = "./dataset"
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

def prepare_dataset():
    # 清理旧数据
    if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
    os.makedirs(os.path.join(OUTPUT_DIR, "audio"), exist_ok=True)

    if not os.path.exists(EXPORT_FILE):
        print(f"❌ 找不到导出文件 {EXPORT_FILE}")
        sys.exit(1)

    with open(EXPORT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = []
    print(f"🔍 [诊断模式] 开始检查 {len(data)} 个任务...")
    print(f"📂 音频源目录: {AUDIO_DIR}")

    for task in data:
        task_id = task.get('id')
        print(f"\n📋 --- 检查任务 Task {task_id} ---")
        
        # 1. 检查音频文件
        audio_url = task.get('data', {}).get('audio', '')
        if not audio_url: 
            print(f"   ❌ 失败: 没有音频 URL")
            continue
            
        decoded_url = urllib.parse.unquote(audio_url)
        fname = os.path.basename(decoded_url.split("?d=")[-1] if "?d=" in decoded_url else decoded_url)
        audio_path = os.path.join(AUDIO_DIR, fname)
        
        if not os.path.exists(audio_path):
            print(f"   ❌ 失败: 音频文件不存在 ({audio_path})")
            continue
        else:
            print(f"   ✅ 音频文件存在: {fname}")

        # 2. 检查标注
        if not task.get('annotations'):
            print(f"   ❌ 失败: 这个任务没有任何标注 (Annotations 为空)")
            continue

        valid_count_in_task = 0
        try:
            # 加载音频获取时长
            y, sr = librosa.load(audio_path, sr=16000)
            audio_len_sec = len(y) / sr

            for ann in task.get('annotations', []):
                results = ann.get('result', [])
                if not results:
                    print(f"   ⚠️ 警告: 标注结果 (result) 是空的")
                
                for i, res in enumerate(results):
                    r_type = res.get('type')
                    print(f"   🧐 [Result {i}] 类型: {r_type}")
                    
                    # 我们只关心 'textarea' (文本转写)
                    if r_type != 'textarea':
                        print(f"      -> 跳过 (原因: 我们需要 'textarea' 类型来训练 Whisper，而这是 '{r_type}')")
                        continue
                    
                    # 检查文本内容
                    text_val = res.get('value', {}).get('text', [])
                    text = text_val[0].strip() if text_val else ""
                    print(f"      -> 文本内容: '{text}'")
                    
                    if not text:
                        print(f"      ❌ 失败: 文本是空的")
                        continue
                    if "正在转写" in text or "在此输入" in text:
                        print(f"      ❌ 失败: 文本包含默认占位符")
                        continue
                        
                    # 检查时间戳
                    start = res['value'].get('start', 0)
                    end = res['value'].get('end', audio_len_sec)
                    duration = end - start
                    print(f"      -> 时间段: {start:.2f}s - {end:.2f}s (时长: {duration:.2f}s)")
                    
                    if duration < 0.1:
                        print(f"      ❌ 失败: 片段太短 (<0.1s)")
                        continue

                    # 一切正常，保存切片
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    y_chunk = y[start_sample:end_sample]
                    
                    chunk_name = f"task{task_id}_{res['id']}.wav"
                    save_path = os.path.join(OUTPUT_DIR, "audio", chunk_name)
                    sf.write(save_path, y_chunk, sr)
                    
                    metadata.append({"file_name": f"audio/{chunk_name}", "sentence": text})
                    valid_count_in_task += 1
                    print(f"      ✅ 成功提取！")

        except Exception as e:
            print(f"   ❌ 处理异常: {e}")

    # 总结
    print("\n" + "="*30)
    if metadata:
        pd.DataFrame(metadata).to_csv(METADATA_PATH, index=False)
        print(f"🎉 最终成功准备了 {len(metadata)} 条数据！")
    else:
        print("🛑 致命错误: 有效数据为 0。请根据上方的 '❌ 失败' 提示去 Label Studio 修改标注。")
        sys.exit(1)

if __name__ == "__main__":
    prepare_dataset()
