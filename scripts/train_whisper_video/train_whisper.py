import os
import sys
import torch
import evaluate
import librosa
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

sys.stdout.reconfigure(line_buffering=True)

# === 🛡️ 路径与设备安全配置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "whisper-finetuned-model")

# 🔥 核心修改：优先使用离线模型
OFFLINE_MODEL_PATH = "/app/models/whisper"
if os.path.exists(os.path.join(OFFLINE_MODEL_PATH, "config.json")):
    print(f"✅ 检测到离线模型，使用: {OFFLINE_MODEL_PATH}")
    MODEL_NAME = OFFLINE_MODEL_PATH
else:
    print("⚠️ 未找到离线模型，将尝试从 HuggingFace 下载 openai/whisper-small")
    MODEL_NAME = "openai/whisper-small"

# 智能设备检测
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print(f"🚀 检测到 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ 降级为 CPU 模式")
# ========================================

# P40/离线 补丁
os.environ["HF_DATASETS_OFFLINE"] = "0"
sys.modules['torchcodec'] = None 
from datasets import config, load_dataset
config.USE_TORCHCODEC = False

from transformers import (
    WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)

def main():
    metadata_path = os.path.join(DATASET_DIR, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"❌ 找不到数据集 {metadata_path}")
        sys.exit(1)

    print("🚀 加载数据集...")
    dataset = load_dataset("csv", data_files=metadata_path)
    dataset = dataset["train"].train_test_split(test_size=0.1)

    print(f"🧠 初始化模型: {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="Chinese", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="Chinese", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    def prepare_dataset_manual(batch):
        audio_path = os.path.join(DATASET_DIR, batch["file_name"])
        speech_array, _ = librosa.load(audio_path, sr=16000)
        batch["input_features"] = processor.feature_extractor(speech_array, sampling_rate=16000).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    print("📊 处理特征...")
    dataset = dataset.map(prepare_dataset_manual, remove_columns=["file_name", "sentence"], num_proc=1)

    @dataclass
    class DataCollator:
        processor: Any
        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        max_steps=500,
        gradient_checkpointing=False,
        fp16=False,
        use_cpu=not USE_CUDA,
        eval_strategy="steps",     
        predict_with_generate=True,
        save_steps=100, eval_steps=100, logging_steps=10,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollator(processor),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("🔥 开始训练...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"🎉 训练完成！保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
