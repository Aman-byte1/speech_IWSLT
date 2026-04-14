# ============================================================
# Fine-tune facebook/mms-1b-all on ALL collected languages
# GPU: NVIDIA A40 (48GB VRAM)
#
# Usage:
#   python finetune_mms_all.py                          # all languages
#   python finetune_mms_all.py --langs amh swh fra      # specific languages
#   python finetune_mms_all.py --resume                 # skip already-done
#   python finetune_mms_all.py --push_to_hub            # push each model to HF
#   python finetune_mms_all.py --epochs 5 --batch_size 8
#   python finetune_mms_all.py --dry_run                # just print config
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Block torchcodec from being imported even if partially installed
import sys
sys.modules["torchcodec"] = None
sys.modules["torchcodec.decoders"] = None
sys.modules["datasets.features._torchcodec"] = None

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
# Maps: config_key -> {
#   "mms_code":  ISO 639-3 code used by MMS adapters,
#   "hf_repo":   HuggingFace dataset repo,
#   "subset":    HF dataset config/subset name (None if single-config),
#   "text_col":  text column name in the dataset,
#   "name":      human-readable name,
# }
# =====================================================================

LANGUAGES = {
    # ── Multi-config repo: african_speech_dataset_new_uncleaned ──────
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
        "mms_code": "arz",
        "hf_repo": "amanuelbyte/african_speech_dataset_new_uncleaned",
        "subset": "arz",
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
    # ── Multi-config repo: african_speech_clean ──────────────────────
    "swh": {
        "mms_code": "swh",
        "hf_repo": "amanuelbyte/african_speech_clean",
        "subset": "swahili",
        "text_col": "text",
        "name": "Swahili",
    },
    "yor": {
        "mms_code": "yor",
        "hf_repo": "amanuelbyte/african_speech_clean",
        "subset": "yoruba",
        "text_col": "text",
        "name": "Yoruba",
    },
    # ── Standalone per-language repos ────────────────────────────────
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
    # ── Pending data collection / New Repos ───────────
    "fra": {
        "mms_code": "fra",
        "hf_repo": "amanuelbyte/african_speech_dataset_fra",
        "subset": "fra_Latn",
        "text_col": "text",
        "name": "French",
    },
    "spa": {
        "mms_code": "spa",
        "hf_repo": "amanuelbyte/african_speech_dataset_spa",
        "subset": "spa_Latn",
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

# =====================================================================
# Defaults
# =====================================================================
MODEL_NAME = "facebook/mms-1b-all"
SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30
OUTPUT_ROOT = "./mms-finetuned"
DEFAULT_MAX_SAMPLES = 23600  # balance all langs to smallest full dataset (Somali)

# =====================================================================
# Logging
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# =====================================================================
# Data Collator
# =====================================================================
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


# =====================================================================
# Helper Functions
# =====================================================================

def load_and_prepare_dataset(
    hf_repo: str,
    text_col: str,
    subset: Optional[str] = None,
    sampling_rate: int = SAMPLING_RATE,
    max_audio_seconds: int = MAX_AUDIO_SECONDS,
    test_size: float = 0.1,
    num_proc: int = 4,
    max_samples: Optional[int] = None,
) -> DatasetDict:
    """Load a HuggingFace dataset and prepare it for CTC training."""
    
    subset_str = f" (subset={subset})" if subset else ""
    log.info(f"  Loading dataset: {hf_repo}{subset_str}")
    
    try:
        ds_iterable = load_dataset(hf_repo, subset, split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        log.warning(f"  Failed to load 'train' split, trying without split: {e}")
        ds_iterable_dict = load_dataset(hf_repo, subset, streaming=True, trust_remote_code=True)
        # Just grab the first available split (usually 'train')
        first_split = list(ds_iterable_dict.keys())[0]
        ds_iterable = ds_iterable_dict[first_split]
    
    if max_samples:
        log.info(f"  Streaming and taking first {max_samples} samples...")
        ds_iterable = ds_iterable.take(max_samples)
    else:
        log.info("  Streaming entire dataset (no max_samples provided)...")

    log.info("  Downloading/caching streamed samples to local disk...")
    from datasets import Dataset
    def gen():
        for ex in ds_iterable:
            yield ex
    
    ds = Dataset.from_generator(gen)
    
    log.info(f"  Loaded {len(ds)} samples into memory. Columns: {ds.column_names}")
    
    # Ensure we have the expected columns
    if "audio" not in ds.column_names:
        # Try common audio column names
        audio_candidates = ["wav", "speech", "audio_path"]
        for col in audio_candidates:
            if col in ds.column_names:
                ds = ds.rename_column(col, "audio")
                log.info(f"  Renamed '{col}' -> 'audio'")
                break
        else:
            raise ValueError(
                f"No audio column found. Available: {ds.column_names}"
            )
    
    # Normalize text column
    if text_col != "text" and text_col in ds.column_names:
        ds = ds.rename_column(text_col, "text")
    elif "text" not in ds.column_names:
        text_candidates = [
            "transcription", "sentence", "transcript", "utt", "normalized_text"
        ]
        for col in text_candidates:
            if col in ds.column_names:
                ds = ds.rename_column(col, "text")
                log.info(f"  Renamed '{col}' -> 'text'")
                break
        else:
            raise ValueError(
                f"No text column found. Available: {ds.column_names}"
            )
    
    # Filter out empty text
    ds = ds.filter(
        lambda x: x["text"] is not None and len(str(x["text"]).strip()) > 1,
        num_proc=num_proc,
    )
    log.info(f"  After text filter: {len(ds)} samples")
    
    # Cast audio to target sampling rate
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    
    # Train/eval split
    split_ds = ds.train_test_split(test_size=test_size, seed=42)
    log.info(f"  Train: {len(split_ds['train'])}, Eval: {len(split_ds['test'])}")
    
    return split_ds


def preprocess_for_ctc(
    dataset: DatasetDict,
    processor: Wav2Vec2Processor,
    sampling_rate: int = SAMPLING_RATE,
    max_audio_seconds: int = MAX_AUDIO_SECONDS,
    num_proc: int = 4,
):
    """Tokenize text and extract audio features for CTC training."""
    
    def prepare_batch(batch):
        audio = batch["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="np",
        )
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(inputs.input_values[0])
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch
    
    max_input_len = sampling_rate * max_audio_seconds
    processed = {}
    
    for split in ["train", "test"]:
        log.info(f"  Preprocessing {split} set...")
        ds = dataset[split].map(
            prepare_batch,
            remove_columns=dataset[split].column_names,
            num_proc=num_proc,
        )
        # Filter out audio longer than max
        ds = ds.filter(lambda x: x["input_length"] < max_input_len)
        processed[split] = ds
        log.info(f"  {split}: {len(ds)} samples (after length filter)")
    
    return processed["train"], processed["test"]


def finetune_one_language(
    lang_key: str,
    lang_cfg: dict,
    args: argparse.Namespace,
    results_log: list,
):
    """Fine-tune MMS on a single language and save the model."""
    
    mms_code = lang_cfg["mms_code"]
    hf_repo = lang_cfg["hf_repo"]
    subset = lang_cfg.get("subset")
    text_col = lang_cfg["text_col"]
    lang_name = lang_cfg["name"]
    
    output_dir = os.path.join(args.output_root, f"mms-{lang_key}-finetuned")
    
    # Skip if already done (--resume mode)
    done_marker = os.path.join(output_dir, "training_done.json")
    if args.resume and os.path.exists(done_marker):
        log.info(f"  ⏭  Already trained (found {done_marker}). Skipping.")
        with open(done_marker) as f:
            prev = json.load(f)
        results_log.append(prev)
        return
    
    start_time = time.time()
    
    # ── 1. Load dataset ─────────────────────────────────────────
    log.info(f"  [1/6] Loading dataset from {hf_repo} ...")
    try:
        split_ds = load_and_prepare_dataset(
            hf_repo=hf_repo,
            text_col=text_col,
            subset=subset,
            sampling_rate=SAMPLING_RATE,
            max_audio_seconds=MAX_AUDIO_SECONDS,
            max_samples=args.max_samples,
            num_proc=args.num_proc,
        )
    except Exception as e:
        log.error(f"  ❌ Failed to load dataset for {lang_name}: {e}")
        results_log.append({
            "lang": lang_key,
            "name": lang_name,
            "status": "FAILED",
            "error": str(e),
        })
        return
    
    # ── 2. Load processor & model ────────────────────────────────
    log.info(f"  [2/6] Loading MMS processor + model (adapter={mms_code}) ...")
    
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    processor.tokenizer.set_target_lang(mms_code)
    
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        ignore_mismatched_sizes=True,
    )
    model.load_adapter(mms_code)
    
    # Freeze the feature encoder (CNN) — only fine-tune transformer + adapter
    model.freeze_feature_encoder()
    
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Total params: {total_p:,} | Trainable: {train_p:,} ({100*train_p/total_p:.2f}%)")
    
    # ── 3. Preprocess ────────────────────────────────────────────
    log.info(f"  [3/6] Preprocessing audio + text ...")
    try:
        train_dataset, eval_dataset = preprocess_for_ctc(
            split_ds, processor,
            sampling_rate=SAMPLING_RATE,
            max_audio_seconds=MAX_AUDIO_SECONDS,
            num_proc=args.num_proc,
        )
    except Exception as e:
        log.error(f"  ❌ Preprocessing failed for {lang_name}: {e}")
        results_log.append({
            "lang": lang_key,
            "name": lang_name,
            "status": "FAILED",
            "error": str(e),
        })
        return
    
    if len(train_dataset) < 10:
        log.warning(f"  ⚠ Only {len(train_dataset)} train samples — skipping {lang_name}.")
        results_log.append({
            "lang": lang_key,
            "name": lang_name,
            "status": "SKIPPED",
            "reason": f"Too few samples ({len(train_dataset)})",
        })
        return
    
    # ── 4. Setup metrics ─────────────────────────────────────────
    log.info(f"  [4/6] Setting up WER/CER metrics ...")
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
    
    # ── 5. Configure training ────────────────────────────────────
    log.info(f"  [5/6] Configuring Trainer ...")
    
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    
    # Adaptive steps based on dataset size
    total_train_steps = (
        len(train_dataset) // (args.batch_size * args.grad_accum) * args.epochs
    )
    eval_save_steps = max(100, min(500, total_train_steps // 5))
    warmup = max(50, min(500, total_train_steps // 10))
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=2,
        logging_steps=max(10, eval_save_steps // 5),
        learning_rate=args.lr,
        warmup_steps=warmup,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",
        logging_dir=os.path.join(output_dir, "logs"),
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    
    # ── 6. Train ─────────────────────────────────────────────────
    log.info(f"  [6/6] 🚀 Training {lang_name} ({lang_key}) ...")
    log.info(f"        Train samples: {len(train_dataset)}")
    log.info(f"        Eval samples:  {len(eval_dataset)}")
    log.info(f"        Epochs: {args.epochs}")
    log.info(f"        Batch: {args.batch_size} x {args.grad_accum} grad_accum")
    log.info(f"        LR: {args.lr}")
    log.info(f"        Est. steps: {total_train_steps}")
    
    try:
        train_result = trainer.train()
    except Exception as e:
        log.error(f"  ❌ Training failed for {lang_name}: {e}")
        results_log.append({
            "lang": lang_key,
            "name": lang_name,
            "status": "FAILED",
            "error": str(e),
        })
        return
    
    # ── Save model ───────────────────────────────────────────────
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    
    # ── Final evaluation ─────────────────────────────────────────
    log.info(f"  Final evaluation for {lang_name} ...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    elapsed = time.time() - start_time
    
    result = {
        "lang": lang_key,
        "name": lang_name,
        "mms_code": mms_code,
        "status": "SUCCESS",
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "final_wer": round(eval_metrics.get("eval_wer", -1), 4),
        "final_cer": round(eval_metrics.get("eval_cer", -1), 4),
        "train_loss": round(train_metrics.get("train_loss", -1), 4),
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Write done marker for --resume
    with open(done_marker, "w") as f:
        json.dump(result, f, indent=2)
    
    results_log.append(result)
    
    log.info(f"  ✅ {lang_name}: WER={result['final_wer']}, CER={result['final_cer']}, "
             f"Time={elapsed/60:.1f}min")
    
    # ── Push to Hub (optional) ───────────────────────────────────
    if args.push_to_hub:
        hub_repo = f"{args.hub_org}/mms-{lang_key}-finetuned"
        log.info(f"  Pushing to Hub: {hub_repo} ...")
        try:
            trainer.push_to_hub(hub_repo)
            processor.push_to_hub(hub_repo)
            log.info(f"  📤 Pushed {lang_name} to {hub_repo}")
        except Exception as e:
            log.warning(f"  ⚠ Failed to push {lang_name}: {e}")
    
    # Free GPU memory
    del model, trainer, train_dataset, eval_dataset
    torch.cuda.empty_cache()


def print_summary(results: list):
    """Print a summary table of all language results."""
    print("\n" + "=" * 90)
    print("TRAINING SUMMARY")
    print("=" * 90)
    print(f"{'Lang':<6} {'Name':<20} {'Status':<10} {'Train':<8} {'Eval':<8} "
          f"{'WER':<10} {'CER':<10} {'Time(m)':<10}")
    print("-" * 90)
    
    for r in results:
        status = r.get("status", "?")
        if status == "SUCCESS":
            print(f"{r['lang']:<6} {r['name']:<20} {'✅':<10} "
                  f"{r.get('train_samples', '?'):<8} {r.get('eval_samples', '?'):<8} "
                  f"{r.get('final_wer', '?'):<10} {r.get('final_cer', '?'):<10} "
                  f"{r.get('elapsed_seconds', 0)/60:<10.1f}")
        elif status == "SKIPPED":
            print(f"{r['lang']:<6} {r.get('name', '?'):<20} {'⏭ SKIP':<10} "
                  f"{'—':<8} {'—':<8} {'—':<10} {'—':<10} {'—':<10}")
        else:
            err = r.get("error", r.get("reason", "?"))[:40]
            print(f"{r['lang']:<6} {r.get('name', '?'):<20} {'❌ FAIL':<10} "
                  f"{'—':<8} {'—':<8} {'—':<10} {'—':<10} {err}")
    
    print("=" * 90)
    
    success = sum(1 for r in results if r.get("status") == "SUCCESS")
    total = len(results)
    print(f"Completed: {success}/{total} languages")
    
    if success > 0:
        avg_wer = np.mean([
            r["final_wer"] for r in results
            if r.get("status") == "SUCCESS" and r.get("final_wer", -1) >= 0
        ])
        avg_cer = np.mean([
            r["final_cer"] for r in results
            if r.get("status") == "SUCCESS" and r.get("final_cer", -1) >= 0
        ])
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Average CER: {avg_cer:.4f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune facebook/mms-1b-all on all collected African speech datasets"
    )
    
    # Language selection
    p.add_argument(
        "--langs", nargs="+", default=None,
        help=f"Languages to fine-tune (default: all). Choices: {list(LANGUAGES.keys())}",
    )
    p.add_argument(
        "--exclude", nargs="+", default=[],
        help="Languages to exclude from training",
    )
    
    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=3, help="Training epochs per language")
    p.add_argument("--batch_size", type=int, default=16, help="Per-device train batch size")
    p.add_argument("--eval_batch_size", type=int, default=8, help="Per-device eval batch size")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    p.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                    help="Max samples per language (default: 23600 to balance datasets)")
    
    # I/O
    p.add_argument("--output_root", type=str, default=OUTPUT_ROOT,
                    help="Root directory for all fine-tuned models")
    p.add_argument("--num_proc", type=int, default=4,
                    help="Number of processes for data preprocessing")
    
    # Hub
    p.add_argument("--push_to_hub", action="store_true", help="Push each model to HF Hub")
    p.add_argument("--hub_org", type=str, default="amanuelbyte",
                    help="HuggingFace org/user for push")
    
    # Workflow
    p.add_argument("--resume", action="store_true",
                    help="Skip languages that already have training_done.json")
    p.add_argument("--dry_run", action="store_true",
                    help="Print config and exit without training")
    
    return p.parse_args()


# =====================================================================
# MAIN
# =====================================================================
def main():
    args = parse_args()
    
    # Resolve language list
    if args.langs:
        lang_keys = [l for l in args.langs if l in LANGUAGES]
        unknown = [l for l in args.langs if l not in LANGUAGES]
        if unknown:
            log.warning(f"Unknown language codes (ignored): {unknown}")
    else:
        lang_keys = list(LANGUAGES.keys())
    
    # Apply exclusions
    lang_keys = [l for l in lang_keys if l not in args.exclude]
    
    print("=" * 70)
    print("  MMS-1b-all  ·  Multilingual Fine-Tuning Pipeline")
    print("=" * 70)
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Languages:   {len(lang_keys)} → {lang_keys}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch:       {args.batch_size} × {args.grad_accum} (eff: {args.batch_size * args.grad_accum})")
    print(f"  LR:          {args.lr}")
    print(f"  Output:      {args.output_root}")
    print(f"  Resume:      {args.resume}")
    print(f"  Push to Hub: {args.push_to_hub}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:         {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
    else:
        print("  ⚠  No GPU detected — training will be very slow!")
    print("=" * 70)
    
    if args.dry_run:
        print("\n[DRY RUN] Configuration printed. Exiting.")
        for k in lang_keys:
            cfg = LANGUAGES[k]
            sub = cfg.get('subset') or '—'
            print(f"  {k:<5} → mms={cfg['mms_code']:<5}  repo={cfg['hf_repo']}  subset={sub}")
        return
    
    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)
    
    # Save run config
    run_config = {
        "model": MODEL_NAME,
        "languages": lang_keys,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_samples": args.max_samples,
        "started_at": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_root, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    
    # ── Train each language ──────────────────────────────────────
    results_log = []
    
    for i, lang_key in enumerate(lang_keys, 1):
        lang_cfg = LANGUAGES[lang_key]
        print("\n" + "━" * 70)
        log.info(f"[{i}/{len(lang_keys)}]  {lang_cfg['name']} ({lang_key})  "
                 f"adapter={lang_cfg['mms_code']}")
        print("━" * 70)
        
        finetune_one_language(lang_key, lang_cfg, args, results_log)
    
    # ── Summary ──────────────────────────────────────────────────
    print_summary(results_log)
    
    # Save full results
    results_path = os.path.join(args.output_root, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(results_log, f, indent=2)
    log.info(f"Full results saved to: {results_path}")


if __name__ == "__main__":
    main()
