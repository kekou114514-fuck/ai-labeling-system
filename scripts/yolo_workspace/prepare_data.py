import json
import os
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
import urllib.parse  # 用于处理URL编码

# === ⚙️ 路径配置 ===
EXPORT_FILE = "./project_export.json" 
AUDIO_DIR = "../测试数据1/音频"
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
    print(f"✂️  开始处理 {len(data)} 个任务...")

    for task in tqdm(data):
        # --- 针对 Local Storage 结构的路径解析 ---
        # 你的数据结构中路径在 task['data']['audio']
        audio_url = task.get('data', {}).get('audio', '')
        if not audio_url:
            print(f"⚠️ 任务 {task.get('id')} 缺少音频路径，跳过")
            continue

        # 1. 解码 URL (处理 %E6%B5%8B%E8%AF%95 等字符)
        decoded_url = urllib.parse.unquote(audio_url)
        
        # 2. 提取文件名 (从 URL 中切分出文件名)
        fname = os.path.basename(decoded_url)
        # 处理可能带有的参数
        if '?' in fname:
            fname = fname.split('?')[0]

        # 3. 寻找音频文件
        audio_path = os.path.join(AUDIO_DIR, fname)
        
        # 如果直接找不到，尝试在子目录深度搜索
        if not os.path.exists(audio_path):
            found = False
            for root, _, files in os.walk(AUDIO_DIR):
                if fname in files:
                    audio_path = os.path.join(root, fname)
                    found = True
                    break
            if not found:
                # print(f"⚠️ 找不到物理文件: {fname}")
                continue

        try:
            audio = AudioSegment.from_file(audio_path)
            for ann in task.get('annotations', []):
                for res in ann.get('result', []):
                    # 只处理文本标注
                    if res.get('type') == 'textarea':
                        text_list = res.get('value', {}).get('text', [])
                        if not text_list: continue
                        
                        text = text_list[0].strip()
                        # 过滤掉空的或者未修改的占位符
                        if not text or "正在转写" in text:
                            continue
                            
                        start_ms = res['value']['start'] * 1000
                        end_ms = res['value']['end'] * 1000
                        
                        # 生成唯一的切片文件名
                        chunk_name = f"task{task['id']}_{res['id']}.wav"
                        save_path = os.path.join(OUTPUT_DIR, "audio", chunk_name)
                        
                        # 执行切分
                        audio[start_ms:end_ms].export(save_path, format="wav")
                        
                        metadata.append({
                            "file_name": f"audio/{chunk_name}",
                            "sentence": text
                        })
        except Exception as e:
            print(f"⚠️ 处理任务 {task.get('id')} 失败: {e}")

    if metadata:
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
        print(f"✅ 成功切分 {len(metadata)} 条有效数据！")
        print(f"📂 数据集已就绪，可以开始训练。")
    else:
        print("❌ 未提取到任何有效数据。")
        print("💡 请检查：1. 标注是否已 Submit；2. 文本框是否仍显示‘AI 正在转写...’")

if __name__ == "__main__":
    prepare_dataset()
