import os
os.environ["HF_DATASETS_USE_TORCHCODEC"] = "0"

from datasets import load_dataset, Audio
import datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
# 1. Load your NEW Translated Dataset
dataset = load_dataset("amanuelbyte/try-task-translation-african-amh-en", split="train")

# Cast audio to 16kHz (Whisper requires this)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Create a train/test split for evaluation
dataset = dataset.train_test_split(test_size=0.1)

# 2. Load Processor
# Set source language to amharic and task to translate
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="amharic", task="translate")

# 3. Data Preparation Function (UPDATED)
def prepare_dataset(batch):
    # Load and resample audio data
    audio = batch["audio"]
    
    # Compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # Encode TARGET translation to label ids (Notice we use batch["translation"] now)
    batch["labels"] = processor.tokenizer(batch["translation"]).input_ids
    return batch

# Map the preparation function (this will take a little time to process the audio)
print("Extracting audio features and tokenizing translations...")
encoded_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)

# 4. Define Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # If bos token is appended in previous tokenization step, cut bos token here
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 5. Load Model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# 6. Define Training Arguments (Optimized for A40 GPU)
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-amharic-to-english-translation",  
    per_device_train_batch_size=16, 
    gradient_accumulation_steps=2,  
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True, 
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer", 
    greater_is_better=False,
    push_to_hub=False,
)

# 7. Initialize Trainer and Train
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

print("Starting Whisper fine-tuning...")
trainer.train()