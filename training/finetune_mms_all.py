# ============================================================
# Fine-tune facebook/mms-1b-all on ALL collected languages
# GPU: NVIDIA A40 (48GB VRAM)
# ============================================================

import os
# Force single-threaded library behavior to prevent PyArrow GIL crashes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import torch
import evaluate
import numpy as np
import datasets
from datasets import load_dataset, Audio, concatenate_datasets, DatasetDict
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# =====================================================================
# LANGUAGE REGISTRY
# =====================================================================
LANGUAGES = {
    "afr": {
        "mms_code": "afr",
        "hf_repo": "amanuelbyte/african_speech_dataset_new_uncleaned",
        "subset": "afr",
        "text_col": "text",
        "name": "Afrikaans",
    },
    "amh": {
        "mms_code": "amh",
        "hf_repo": "amanuelbyte/african_speech_dataset_new_uncleaned",
        "subset": "amh",
        "text_col": "text",
        "name": "Amharic",
    },
    "arz": {
        "mms_code": "ara",
        "hf_repo": "amanuelbyte/african_speech_dataset_new_uncleaned",
        "subset": "arz_clean",
        "text_col": "text",
        "name": "Egyptian Arabic",
    },
    "hau": {
        "mms_code": "hau",
        "hf_repo": "amanuelbyte/african_speech_dataset_new_uncleaned",
        "subset": "hau",
        "text_col": "text",
        "name": "Hausa",
    },
    "som": {
        "mms_code": "som",
        "hf_repo": "amanuelbyte/african_speech_dataset_new_uncleaned",
        "subset": "som",
        "text_col": "text",
        "name": "Somali",
    },
    "swh": {
        "mms_code": "swh",
        "hf_repo": "amanuelbyte/african_speech_clean",
        "subset": "swahili_clean",
        "text_col": "text",
        "name": "Swahili",
    },
    "yor": {
        "mms_code": "yor",
        "hf_repo": "amanuelbyte/african_speech_clean",
        "subset": "yoruba_clean",
        "text_col": "text",
        "name": "Yoruba",
    },
    "arb": {
        "mms_code": "ara",
        "hf_repo": "amanuelbyte/african_speech_dataset_arb",
        "subset": "arb_Arab",
        "text_col": "text",
        "name": "Arabic (MSA)",
    },
    "zul": {
        "mms_code": "zul",
        "hf_repo": "amanuelbyte/african_speech_dataset_zul",
        "subset": "zul",
        "text_col": "text",
        "name": "Zulu",
    },
    "fra": {
        "mms_code": "fra",
        "hf_repo": "amanuelbyte/african_speech_dataset_fra",
        "subset": "fra_Latn_clean",
        "text_col": "text",
        "name": "French",
    },
    "spa": {
        "mms_code": "spa",
        "hf_repo": "amanuelbyte/african_speech_dataset_spa",
        "subset": "spa_Latn_clean",
        "text_col": "text",
        "name": "Spanish",
    },
    "por": {
        "mms_code": "por",
        "hf_repo": "amanuelbyte/african_speech_dataset_por",
        "subset": "por_Latn",
        "text_col": "text",
        "name": "Portuguese",
    },
}

MODEL_NAME = "facebook/mms-1b-all"
SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30
OUTPUT_ROOT = "./mms-finetuned"
DEFAULT_MAX_SAMPLES = 10000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def load_and_prepare_dataset(
    hf_repo: str,
    text_col: str,
    subset: Optional[str] = None,
    sampling_rate: int = SAMPLING_RATE,
    max_audio_seconds: int = MAX_AUDIO_SECONDS,
    test_size: float = 0.1,
    num_proc: int = 0,
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
) -> DatasetDict:
    subset_str = f" (subset={subset})" if subset else ""
    log.info(f"  Loading dataset: {hf_repo}{subset_str}")
    
    # Use streaming=False to avoid PyArrow threading crashes entirely
    try:
        ds = load_dataset(hf_repo, subset, split="train", streaming=False, token=token)
    except Exception as e:
        log.warning(f"  Failed split load, trying without split: {e}")
        ds_dict = load_dataset(hf_repo, subset, streaming=False, token=token)
        first_split = list(ds_dict.keys())[0]
        ds = ds_dict[first_split]
    
    if max_samples and len(ds) > max_samples:
        log.info(f"  Selecting {max_samples} samples...")
        ds = ds.select(range(max_samples))
    
    # Columns cleanup
    if "audio" not in ds.column_names:
        audio_candidates = ["wav", "speech", "audio_path"]
        for col in audio_candidates:
            if col in ds.column_names:
                ds = ds.rename_column(col, "audio")
                break
                
    if text_col != "text" and text_col in ds.column_names:
        ds = ds.rename_column(text_col, "text")
    elif "text" not in ds.column_names:
        text_candidates = ["transcription", "sentence", "transcript", "normalized_text"]
        for col in text_candidates:
            if col in ds.column_names:
                ds = ds.rename_column(col, "text")
                break

    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.filter(lambda x: x["text"] is not None and len(str(x["text"]).strip()) > 1)
    
    return ds.train_test_split(test_size=test_size, seed=42)

def preprocess_for_ctc(dataset, processor, sampling_rate, max_audio_seconds, num_proc=0):
    import soundfile as sf
    import librosa
    import io
    
    def prepare_batch(batch):
        audio = batch["audio"]
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            with open(audio["path"], "rb") as f:
                audio_bytes = f.read()
        try:
            with io.BytesIO(audio_bytes) as f:
                array, sr = sf.read(f, dtype="float32")
        except:
            with io.BytesIO(audio_bytes) as f:
                array, sr = librosa.load(f, sr=None, dtype="float32")
        
        if sr != sampling_rate:
            array = librosa.resample(y=array, orig_sr=sr, target_sr=sampling_rate)

        try:
            inputs = processor(
                array,
                sampling_rate=sampling_rate,
                return_tensors="np",
            )
            batch["input_values"] = inputs.input_values[0]
            batch["input_length"] = len(inputs.input_values[0])
            batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        except Exception as e:
            log.warning(f"  ⚠ Failed to process sample: {e}")
            batch["input_values"] = [0.0] * 100
            batch["input_length"] = 100
            batch["labels"] = [0]
        return batch

    max_input_len = sampling_rate * max_audio_seconds
    processed = {}
    for split in ["train", "test"]:
        ds = dataset[split].map(prepare_batch, remove_columns=dataset[split].column_names, num_proc=num_proc)
        ds = ds.filter(lambda x: x["input_length"] < max_input_len and (x["input_length"] // 320) > len(x["labels"]))
        processed[split] = ds
    return processed["train"], processed["test"]

def finetune_one_language(lang_key, lang_cfg, args, results_log):
    mms_code = lang_cfg["mms_code"]
    hf_repo = lang_cfg["hf_repo"]
    subset = lang_cfg.get("subset")
    text_col = lang_cfg["text_col"]
    lang_name = lang_cfg["name"]
    output_dir = os.path.join(args.output_root, f"mms-{lang_key}-finetuned")
    
    done_marker = os.path.join(output_dir, "training_done.json")
    if args.resume and os.path.exists(done_marker):
        with open(done_marker) as f:
            prev = json.load(f)
        results_log.append(prev)
        return

    log.info(f"--- Processing {lang_name} ---")
    try:
        split_ds = load_and_prepare_dataset(
            hf_repo, text_col, subset, 
            max_samples=args.max_samples,
            token=os.environ.get("HF_TOKEN")
        )
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        processor.tokenizer.set_target_lang(mms_code)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, ctc_loss_reduction="mean", ignore_mismatched_sizes=True)
        model.load_adapter(mms_code)
        model.freeze_feature_encoder()
        
        train_ds, eval_ds = preprocess_for_ctc(split_ds, processor, SAMPLING_RATE, MAX_AUDIO_SECONDS, args.num_proc)
        
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")
        def compute_metrics(pred):
            pred_ids = np.argmax(pred.predictions, axis=-1)
            pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.batch_decode(pred_ids)
            label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
            return {"wer": wer_metric.compute(predictions=pred_str, references=label_str),
                    "cer": cer_metric.compute(predictions=pred_str, references=label_str)}

        num_train_epochs = getattr(args, "epochs", getattr(args, "num_train_epochs", 1))

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=output_dir,
            num_train_epochs=num_train_epochs,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.lr,
                fp16=True,
                save_total_limit=1,
                push_to_hub=False,
                report_to="none"
            ),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=DataCollatorCTCWithPadding(processor=processor),
            compute_metrics=compute_metrics,
        )
        trainer.train()
        eval_metrics = trainer.evaluate()
        
        res = {"lang": lang_key, "name": lang_name, "status": "SUCCESS", "final_wer": eval_metrics["eval_wer"]}
        with open(done_marker, "w") as f: json.dump(res, f)
        results_log.append(res)
        
        if args.push_to_hub:
            trainer.push_to_hub(f"{args.hub_org}/mms-{lang_key}-finetuned")
            
    except Exception as e:
        log.error(f"Failed {lang_name}: {e}")
        results_log.append({"lang": lang_key, "name": lang_name, "status": "FAILED", "error": str(e)})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--max_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output_root", type=str, default="./mms-finetuned")
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument("--hub_org", type=str, default="amanuelbyte")
    args = p.parse_args()
    
    lang_keys = args.langs if args.langs else list(LANGUAGES.keys())
    results = []
    for k in lang_keys:
        finetune_one_language(k, LANGUAGES[k], args, results)
    
    print("\nSUMMARY:")
    for r in results: print(f"{r['lang']}: {r['status']} (WER: {r.get('final_wer', 'N/A')})")

if __name__ == "__main__":
    main()
