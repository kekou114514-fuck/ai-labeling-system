import os
import glob
import json
import argparse
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

# ==========================================
# ⚙️ Docker 适配配置
# ==========================================
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
LS_URL_PREFIX = "/data/local-files/?d=/data/"

# 强制离线，优先使用 Docker 内置模型
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def run_inference(project_type):
    # === P2: 纯音频 ===
    if project_type == '2':
        print("🎧 模式: 项目 2 (纯音频)")
        config = {
            "audio_dir": os.path.join(DATA_ROOT, "audio"),
            # 优先读取您粘贴进去的离线模型
            "model_path": "/app/models/whisper", 
            "output": os.path.join(DATA_ROOT, "outputs/pre_annotations_audio.json")
        }
    # === P3: 视频语音 ===
    elif project_type == '3':
        print("🎬 模式: 项目 3 (视频提取音频)")
        config = {
            "audio_dir": os.path.join(DATA_ROOT, "video_audio"),
            # 如果 P3 有专门微调的模型，可以改这里；默认也用基础模型
            "model_path": "/app/models/whisper",
            "output": os.path.join(DATA_ROOT, "outputs/pre_annotations_video_audio.json")
        }
    else:
        print("❌ 未知项目类型")
        return

    # 1. 检查音频目录
    if not os.path.exists(config['audio_dir']):
        print(f"❌ 找不到音频文件夹: {config['audio_dir']}")
        return

    # 2. 加载模型
    print(f"🧠 加载模型: {config['model_path']}")
    try:
        if os.path.exists(os.path.join(config['model_path'], "config.json")):
            model = WhisperForConditionalGeneration.from_pretrained(config['model_path'])
            processor = WhisperProcessor.from_pretrained(config['model_path'])
        else:
            print("⚠️ 离线模型未找到，尝试联网加载 openai/whisper-small...")
            os.environ["HF_HUB_OFFLINE"] = "0" # 临时开启联网
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"🚀 模型已加载至 {device}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 扫描文件
    extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(config['audio_dir'], ext)))
        audio_files.extend(glob.glob(os.path.join(config['audio_dir'], ext.upper())))

    if not audio_files:
        print(f"❌ 未找到音频文件: {config['audio_dir']}")
        return

    print(f"🎤 开始处理 {len(audio_files)} 个文件...")
    results_list = []

    for audio_path in tqdm(audio_files):
        try:
            # 读取并转写
            speech, _ = librosa.load(audio_path, sr=16000)
            input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            
            with torch.no_grad():
                predicted_ids = model.generate(input_features, language="zh", task="transcribe")
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # 生成相对路径 URL
            rel_path = os.path.relpath(audio_path, DATA_ROOT)
            ls_url = f"{LS_URL_PREFIX}{rel_path}"

            results_list.append({
                "data": {"audio": ls_url},
                "predictions": [{
                    "model_version": "whisper_v1",
                    "result": [{
                        "from_name": "transcription",
                        "to_name": "audio",
                        "type": "textarea",
                        "value": {"text": [transcription]}
                    }]
                }]
            })
        except Exception as e:
            print(f"⚠️ 跳过文件 {os.path.basename(audio_path)}: {e}")

    # 4. 保存
    os.makedirs(os.path.dirname(config['output']), exist_ok=True)
    with open(config['output'], 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)

    print(f"✅ 生成完毕: {config['output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    args = parser.parse_args()
    run_inference(args.project)
