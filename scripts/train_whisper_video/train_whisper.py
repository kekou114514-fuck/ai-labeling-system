import os
import sys
import torch
import evaluate
import librosa
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import datasets
from datasets import load_dataset, DatasetDict

sys.stdout.reconfigure(line_buffering=True)
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "whisper-finetuned-model")
MODEL_PATH = "/app/models/whisper"

from transformers import (
    WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)

def main():
    if not os.path.exists(os.path.join(DATASET_DIR, "metadata.csv")):
        print("❌ 错误：未找到数据文件。")
        sys.exit(1)

    print(f"🚀 加载模型: {MODEL_PATH}")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="Chinese", task="transcribe")
        tokenizer = WhisperTokenizer.from_pretrained(MODEL_PATH, language="Chinese", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"❌ 加载失败: {e}"); sys.exit(1)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # 加载数据
    dataset = load_dataset("csv", data_files=os.path.join(DATASET_DIR, "metadata.csv"), split="train")
    total_samples = len(dataset)
    print(f"📊 数据集总量: {total_samples} 条")

    def prepare_dataset(batch):
        path = os.path.join(DATASET_DIR, batch["file_name"])
        try:
            speech, _ = librosa.load(path, sr=16000)
            batch["input_features"] = processor.feature_extractor(speech, sampling_rate=16000).input_features[0]
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
        except:
            batch["input_features"] = None
        return batch

    dataset = dataset.map(prepare_dataset, num_proc=1).filter(lambda x: x["input_features"] is not None)
    
    # 🔥 核心修正：单样本/少样本策略
    if len(dataset) < 2:
        print("⚠️ 警告：样本极少 (<2)，跳过验证集划分，开启全量过拟合训练模式。")
        dataset = DatasetDict({"train": dataset, "test": dataset}) # test即train，仅为防报错
        eval_strategy = "no"
        save_steps = 10
        logging_steps = 1
    else:
        # 正常划分
        dataset = dataset.train_test_split(test_size=0.1)
        eval_strategy = "steps"
        save_steps = 50
        logging_steps = 10

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
        per_device_train_batch_size=2, 
        learning_rate=1e-5, 
        max_steps=50, # 强制最大步数，避免单样本无限训练
        fp16=torch.cuda.is_available(), 
        logging_steps=logging_steps, 
        save_steps=save_steps, 
        eval_strategy=eval_strategy, # 动态调整验证策略
        report_to=[],
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        args=training_args, model=model, train_dataset=dataset["train"], 
        eval_dataset=dataset["test"] if eval_strategy != "no" else None,
        data_collator=DataCollator(processor), compute_metrics=compute_metrics, tokenizer=processor.feature_extractor
    )

    print("🔥 开始训练...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"🎉 训练完成！新模型已保存。")

if __name__ == "__main__":
    main()
