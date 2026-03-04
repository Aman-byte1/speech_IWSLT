# ============================================================
# Fine-tune facebook/mms-1b-all on Amharic Speech Dataset
# GPU: NVIDIA A40 (48GB VRAM)
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Block torchcodec from being imported even if partially installed
import sys
sys.modules["torchcodec"] = None
sys.modules["torchcodec.decoders"] = None
sys.modules["datasets.features._torchcodec"] = None

import torch
import evaluate
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ----- Configuration -----
MODEL_NAME = "facebook/mms-1b-all"
DATASET_NAME = "amanuelbyte/african_speech_clean"
SUBSET = "amharic"
LANGUAGE_CODE = "amh"
OUTPUT_DIR = "./mms-amharic-finetuned-full"
SAMPLING_RATE = 16000

NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 100

print("=" * 60)
print("Fine-tuning MMS-1b-all for Amharic ASR")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

# ----- 1. Load Dataset -----
print("\n[1/6] Loading dataset...")
dataset = load_dataset(
    DATASET_NAME,
    SUBSET,
    split="train",
    trust_remote_code=True,
)

print(f"Total samples: {len(dataset)}")
print(f"Columns: {dataset.column_names}")

# Cast audio to 16kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

# Split 90/10
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# ----- 2. Load Processor & Model -----
print("\n[2/6] Loading processor and model...")

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
processor.tokenizer.set_target_lang(LANGUAGE_CODE)

model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    ignore_mismatched_sizes=True,
)
model.load_adapter(LANGUAGE_CODE)
model.freeze_feature_encoder()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

# ----- 3. Preprocess -----
print("\n[3/6] Preprocessing dataset...")


def prepare_dataset(batch):
    """Process a single example."""
    audio = batch["audio"]
    
    # audio is already decoded dict: {"array": np.array, "sampling_rate": int, "path": str}
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    )
    batch["input_values"] = inputs.input_values[0]
    batch["input_length"] = len(inputs.input_values[0])

    # Tokenize text
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids

    return batch


# *** num_proc=1 to avoid multiprocessing audio decode issues ***
print("  Processing train set...")
train_dataset = train_dataset.map(
    prepare_dataset,
    remove_columns=train_dataset.column_names,
    num_proc=8,
)

print("  Processing eval set...")
eval_dataset = eval_dataset.map(
    prepare_dataset,
    remove_columns=eval_dataset.column_names,
    num_proc=8,
)

# Filter long audio (>30s)
MAX_INPUT_LENGTH = SAMPLING_RATE * 30
train_dataset = train_dataset.filter(lambda x: x["input_length"] < MAX_INPUT_LENGTH)
eval_dataset = eval_dataset.filter(lambda x: x["input_length"] < MAX_INPUT_LENGTH)
print(f"Train samples (after filter): {len(train_dataset)}")
print(f"Eval samples (after filter): {len(eval_dataset)}")


# ----- 4. Data Collator -----
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )

        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)

# ----- 5. Metrics -----
print("\n[4/6] Setting up metrics...")
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    filtered = [
        (p, l) for p, l in zip(pred_str, label_str) if len(l.strip()) > 0
    ]
    if not filtered:
        return {"wer": 1.0, "cer": 1.0}

    pred_str, label_str = zip(*filtered)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}


# ----- 6. Training -----
print("\n[5/6] Configuring training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    fp16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    report_to="none",
    logging_dir=f"{OUTPUT_DIR}/logs",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

print("\n[6/6] Starting training...")
print("=" * 60)
train_result = trainer.train()

# Save
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Final eval
print("\nFinal evaluation...")
eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)
print(f"Final WER: {eval_metrics['eval_wer']:.4f}")
print(f"Final CER: {eval_metrics['eval_cer']:.4f}")
print(f"\nModel saved to: {OUTPUT_DIR}")
print("Done! 🎉")