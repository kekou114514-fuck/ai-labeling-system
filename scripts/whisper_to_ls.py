import os
import glob
import json
import argparse
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

DATA_ROOT = os.getenv('DATA_ROOT', '/data')
LS_URL_PREFIX = "/data/local-files/?d=/data/"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 幻觉词过滤
HALLUCINATION_PHRASES = ["你好", "字幕", "请订阅", "谢谢观看", "Subtitle", "Copyright"]

def run_inference(project_type):
    # === P2: 纯音频 (通常是 ID 3) ===
    if project_type in ['2', '3']:
        print(f"🎧 模式: 项目 {project_type} (纯音频)")
        config = {
            "audio_dir": os.path.join(DATA_ROOT, "audio"),
            "model_path": "/app/models/whisper", 
            "output": os.path.join(DATA_ROOT, "outputs/pre_annotations_audio.json")
        }
    # === P3: 视频语音 (ID 5 或 6) ===
    # 🔥 核心修改：增加 ID 5 支持
    elif project_type in ['5', '6']:
        print(f"🎬 模式: 项目 {project_type} (视频提取音频)")
        config = {
            "audio_dir": os.path.join(DATA_ROOT, "video_audio"),
            # 优先使用微调后的模型
            "model_path": "/app/scripts/train_whisper_video/whisper-finetuned-model",
            "output": os.path.join(DATA_ROOT, "outputs/pre_annotations_video_audio.json")
        }
    else:
        print(f"❌ 未知项目类型: {project_type}"); return

    # 模型回退逻辑
    if not os.path.exists(config['model_path']):
        print(f"⚠️ 微调模型未找到，使用基础模型: /app/models/whisper")
        config['model_path'] = "/app/models/whisper"

    # 1. 扫描文件
    if not os.path.exists(config['audio_dir']):
        print(f"❌ 目录不存在: {config['audio_dir']}"); return

    extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(config['audio_dir'], ext)))
        audio_files.extend(glob.glob(os.path.join(config['audio_dir'], ext.upper())))

    if not audio_files:
        print(f"❌ 未找到音频: {config['audio_dir']}"); return

    # 2. 加载模型
    print(f"🧠 加载模型: {config['model_path']}")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(config['model_path'])
        processor = WhisperProcessor.from_pretrained(config['model_path'])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    except Exception as e:
        print(f"❌ 加载失败: {e}"); return

    print(f"🎤 开始推理 {len(audio_files)} 个文件...")
    results_list = []

    for audio_path in tqdm(audio_files):
        try:
            speech, _ = librosa.load(audio_path, sr=16000)
            
            # 静音检测
            if np.max(np.abs(speech)) < 0.005:
                transcription = "[静音]"
            else:
                if len(speech) > 16000 * 30: speech = speech[:16000*30] # 只取前30秒
                
                input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(device)
                with torch.no_grad():
                    # 自动检测语言
                    predicted_ids = model.generate(input_features, task="transcribe", no_repeat_ngram_size=2, num_beams=5)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

                # 幻觉过滤
                if transcription in HALLUCINATION_PHRASES or len(transcription) < 2:
                    transcription = f"[可能为噪音] ({transcription})"

            rel_path = os.path.relpath(audio_path, DATA_ROOT)
            ls_url = f"{LS_URL_PREFIX}{rel_path}"

            # 🔥 适配 P3 XML: to_name="audio", from_name="transcription", type="textarea"
            results_list.append({
                "data": {"audio": ls_url},
                "predictions": [{
                    "model_version": "v1",
                    "result": [{
                        "from_name": "transcription", 
                        "to_name": "audio", 
                        "type": "textarea",
                        "value": {"text": [transcription]}
                    }]
                }]
            })
        except: continue

    os.makedirs(os.path.dirname(config['output']), exist_ok=True)
    with open(config['output'], 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    print(f"✅ 生成完毕: {config['output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    args = parser.parse_args()
    run_inference(args.project)
