import os
import sys
import torch
import evaluate
import numpy as np
import librosa  
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# 强制离线模式
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout.reconfigure(line_buffering=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "whisper-finetuned-model")
MODEL_PATH = "/app/models/whisper"

# 🔥 核心修复：明确导入 DatasetDict
import datasets 
from datasets import load_dataset, DatasetDict 

from transformers import (
    WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)

def main():
    if not os.path.exists(os.path.join(DATASET_DIR, "metadata.csv")):
        print("❌ 错误：未找到 dataset/metadata.csv")
        sys.exit(1)

    print(f"🚀 加载模型: {MODEL_PATH}")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="Chinese", task="transcribe")
        tokenizer = WhisperTokenizer.from_pretrained(MODEL_PATH, language="Chinese", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # 加载数据集
    dataset = load_dataset("csv", data_files=os.path.join(DATASET_DIR, "metadata.csv"), split="train")
    
    # 1. 把相对路径转为绝对路径
    def resolve_audio_path(batch):
        batch["audio_path"] = [os.path.join(DATASET_DIR, f) for f in batch["file_name"]]
        return batch
    
    dataset = dataset.map(resolve_audio_path, batched=True)

    # 2. 手动读取音频 (librosa)
    def prepare_dataset(batch):
        path = batch["audio_path"]
        try:
            # 强制重采样到 16k
            speech_array, sampling_rate = librosa.load(path, sr=16000)
            
            # 提取特征
            batch["input_features"] = processor.feature_extractor(
                speech_array, sampling_rate=sampling_rate
            ).input_features[0]
            
            # 编码标签
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
        except Exception as e:
            print(f"⚠️ 读取音频失败 {path}: {e}")
            batch["input_features"] = [] 
            batch["labels"] = []
            
        return batch

    print("📊 预处理数据 (使用 librosa 手动读取)...")
    dataset = dataset.map(prepare_dataset, num_proc=1).filter(lambda x: len(x["input_features"]) > 0)
    
    # 划分验证集
    if len(dataset) > 5:
        dataset = dataset.train_test_split(test_size=0.1)
    else:
        print("⚠️ 数据量较少，跳过验证集划分")
        # 🔥 修复点：现在 DatasetDict 已经导入了，不会报错了
        dataset = DatasetDict({"train": dataset, "test": dataset})

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        def __call__(self, features):
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        max_steps=50, 
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_steps=25,
        report_to=[], 
        remove_unused_columns=False 
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("🔥 开始训练...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"🎉 训练完成！新模型保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
