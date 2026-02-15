#!/usr/bin/env python3
"""
African Speech Data Pipeline - Robust Edition
================================================
Features:
- Checkpoint/Resume capability - continues from where it left off
- Robust error handling with retry logic
- Per-language split organization
- Comprehensive dataset sources
- Progress tracking and logging
"""

import argparse
import sys
import yaml
import os
import logging
import time
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict

# Disable Xet backend to avoid MerkleDB errors on some filesystems
os.environ["HF_HUB_DISABLE_XET"] = "1"

import pandas as pd
import torch
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
from datasets import load_dataset, Dataset, Audio, Value, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import noisereduce as nr
import re

# Setup Logging with file rotation
def setup_logging(log_dir: str = "./logs"):
    """Setup comprehensive logging with both file and console output."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # File handler with rotation
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)

logger = setup_logging()

@dataclass
class ProcessStats:
    """Statistics for a single dataset processing run."""
    dataset_name: str
    language: str
    original_size: int = 0
    final_size: int = 0
    discarded_count: int = 0
    status: str = "pending"  # pending, processing, completed, failed, skipped
    error_message: str = ""
    start_time: str = ""
    end_time: str = ""
    retry_count: int = 0

@dataclass
class PipelineState:
    """Complete state of the pipeline for checkpointing."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    datasets_processed: Dict[str, Dict] = field(default_factory=dict)
    total_original: int = 0
    total_final: int = 0
    languages_completed: List[str] = field(default_factory=list)
    current_language: str = ""

    def save(self, path: str):
        """Save state to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PipelineState':
        """Load state from JSON file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()

class CheckpointManager:
    """Manages checkpointing and resume functionality."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.state_file = os.path.join(checkpoint_dir, "pipeline_state.json")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.state = PipelineState.load(self.state_file)
        logger.info(f"Loaded checkpoint state from {self.state_file}")
        logger.info(f"Previously processed datasets: {len(self.state.datasets_processed)}")

    def is_dataset_processed(self, dataset_name: str) -> bool:
        """Check if a dataset has been successfully processed."""
        if dataset_name in self.state.datasets_processed:
            return self.state.datasets_processed[dataset_name].get('status') == 'completed'
        return False

    def mark_started(self, dataset_name: str, language: str):
        """Mark a dataset as started."""
        self.state.datasets_processed[dataset_name] = {
            'language': language,
            'status': 'processing',
            'start_time': datetime.now().isoformat()
        }
        self.state.current_language = language
        self.save()

    def mark_completed(self, dataset_name: str, original_size: int, final_size: int):
        """Mark a dataset as completed."""
        if dataset_name in self.state.datasets_processed:
            self.state.datasets_processed[dataset_name].update({
                'status': 'completed',
                'original_size': original_size,
                'final_size': final_size,
                'end_time': datetime.now().isoformat()
            })
        else:
            self.state.datasets_processed[dataset_name] = {
                'status': 'completed',
                'original_size': original_size,
                'final_size': final_size,
                'end_time': datetime.now().isoformat()
            }
        self.state.total_original += original_size
        self.state.total_final += final_size
        self.save()

    def mark_failed(self, dataset_name: str, error: str, retry_count: int = 0):
        """Mark a dataset as failed."""
        self.state.datasets_processed[dataset_name] = {
            'status': 'failed',
            'error': error,
            'retry_count': retry_count,
            'end_time': datetime.now().isoformat()
        }
        self.save()

    def mark_skipped(self, dataset_name: str, reason: str = "already_processed"):
        """Mark a dataset as skipped."""
        self.state.datasets_processed[dataset_name] = {
            'status': 'skipped',
            'reason': reason,
            'end_time': datetime.now().isoformat()
        }
        self.save()

    def save(self):
        """Save current state to disk."""
        self.state.save(self.state_file)

    def get_pending_datasets(self, all_datasets: List[Dict]) -> List[Dict]:
        """Get list of datasets that still need processing."""
        pending = []
        for ds in all_datasets:
            ds_name = ds.get('name')
            if not self.is_dataset_processed(ds_name):
                pending.append(ds)
            else:
                logger.info(f"Skipping already processed dataset: {ds_name}")
        return pending

    def reset_dataset(self, dataset_name: str):
        """Reset a dataset's status to allow reprocessing."""
        if dataset_name in self.state.datasets_processed:
            del self.state.datasets_processed[dataset_name]
            self.save()
            logger.info(f"Reset status for dataset: {dataset_name}")

class TextProcessor:
    """Handles text processing and filtering."""

    def __init__(self, config: Dict):
        self.config = config.get('text', {})
        self.models_config = config.get('models', {})
        self.semantic_model = None

        # Load Semantic Model if needed (Lazy loading)
        if 'semantic_model' in self.models_config and self.models_config['semantic_model']:
            try:
                logger.info(f"Loading Semantic Model: {self.models_config['semantic_model']}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.semantic_model = SentenceTransformer(
                    self.models_config['semantic_model'],
                    device=device
                )
            except Exception as e:
                logger.warning(f"Could not load semantic model: {e}. Semantic filtering will be skipped.")

    def basic_clean(self, text: str) -> str:
        """Removes HTML, extra spaces, etc."""
        if not text or not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters but keep language-specific ones
        text = re.sub(r'[^\w\s\u0600-\u06FF\u1200-\u137F\u0800-\u10FF]', '', text)

        return text.lower()

    def filter_rules(self, src_text: str, tgt_text: Optional[str] = None) -> bool:
        """Stage 1: Rule-based filtering (Length, Ratio)."""
        if not src_text:
            return False

        # Length check
        min_chars = self.config.get('min_chars', 3)
        max_chars = self.config.get('max_chars', 300)

        if not (min_chars <= len(src_text) <= max_chars):
            return False

        # Ratio check (if target exists)
        if tgt_text:
            len_src = len(src_text)
            len_tgt = len(tgt_text)

            if len_src == 0 or len_tgt == 0:
                return False

            ratio = max(len_src, len_tgt) / min(len_src, len_tgt)
            max_ratio = self.config.get('max_ratio', 2.5)

            if ratio > max_ratio:
                return False

        return True

    def semantic_filter(self, src_texts: List[str], tgt_texts: List[str]) -> List[bool]:
        """Stage 3: Semantic Filtering using LaBSE or similar."""
        if self.semantic_model is None:
            return [True] * len(src_texts)

        logger.info("Running Semantic Filtering...")
        try:
            embeddings1 = self.semantic_model.encode(
                src_texts, convert_to_tensor=True, show_progress_bar=False
            )
            embeddings2 = self.semantic_model.encode(
                tgt_texts, convert_to_tensor=True, show_progress_bar=False
            )

            import torch.nn.functional as F
            scores = F.cosine_similarity(embeddings1, embeddings2)

            threshold = self.config.get('semantic_threshold', 0.6)
            return (scores >= threshold).cpu().tolist()
        except Exception as e:
            logger.error(f"Semantic filtering failed: {e}")
            return [True] * len(src_texts)


class AudioProcessor:
    """Handles audio processing operations."""

    def __init__(self, config: Dict):
        self.config = config.get('audio', {})
        self.target_sr = self.config.get('sampling_rate', 16000)
        self.min_duration = self.config.get('min_duration', 1.0)
        self.max_duration = self.config.get('max_duration', 30.0)
        self.trim_silence = self.config.get('trim_silence', True)
        self.top_db = self.config.get('top_db', 20)

    def load_audio(self, audio_path: str) -> tuple:
        """Load audio file and return waveform and sample rate."""
        try:
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

            # Trim silence if enabled
            if self.trim_silence and len(y) > 0:
                y, _ = librosa.effects.trim(y, top_db=self.top_db)

            return y, sr
        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {e}")
            return None, 0

    def get_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            return librosa.get_duration(path=audio_path)
        except:
            y, sr = self.load_audio(audio_path)
            if y is not None:
                return len(y) / sr
            return 0

    def is_valid_duration(self, duration: float) -> bool:
        """Check if duration is within valid range."""
        return self.min_duration <= duration <= self.max_duration


class DataPipeline:
    """Main data pipeline with robust checkpointing and error handling."""

    def __init__(self, config_path: str, checkpoint_dir: str = "./checkpoints"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.text_proc = TextProcessor(self.config)
        self.audio_proc = AudioProcessor(self.config)
        self.checkpoint = CheckpointManager(checkpoint_dir)

        self.hf_token = self.config.get('hf_token')
        self.output_org = self.config.get('output_org')

        # Create necessary directories
        self.output_dir = self.config.get('output_dir', './processed_data')
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("African Speech Data Pipeline - Robust Edition")
        logger.info("=" * 60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"HuggingFace Token: {'Provided' if self.hf_token else 'Not Provided'}")
        logger.info(f"Output Organization: {self.output_org or 'Local Only'}")

    def run(self, lang_filter: Optional[str] = None, skip_processing: bool = False,
            dataset_name: Optional[str] = None):
        """
        Run the pipeline with checkpoint support.

        Args:
            lang_filter: Filter datasets by language code
            skip_processing: If True, just copy without processing
            dataset_name: Process only this specific dataset
        """
        datasets_config = self.config.get('datasets', [])

        # Apply filters
        filtered_datasets = []
        for ds_conf in datasets_config:
            # Language filter
            if lang_filter:
                ds_lang = ds_conf.get('language', '').lower()
                if ds_lang != lang_filter.lower():
                    continue

            # Dataset name filter
            if dataset_name:
                if ds_conf.get('name') != dataset_name:
                    continue

            filtered_datasets.append(ds_conf)

        # Get pending datasets (respecting checkpoint)
        pending_datasets = self.checkpoint.get_pending_datasets(filtered_datasets)

        logger.info(f"Total datasets in config: {len(datasets_config)}")
        logger.info(f"Datasets to process: {len(pending_datasets)}")

        # Process each dataset
        for i, ds_conf in enumerate(pending_datasets):
            ds_name = ds_conf.get('name')
            logger.info(f"\n[{i+1}/{len(pending_datasets)}] Processing: {ds_name}")

            # Mark as started
            self.checkpoint.mark_started(ds_name, ds_conf.get('language', 'unknown'))

            try:
                self.process_dataset(ds_conf, skip_processing)
                self.checkpoint.mark_completed(
                    ds_name,
                    ds_conf.get('_original_size', 0),
                    ds_conf.get('_final_size', 0)
                )
                logger.info(f"✓ Successfully processed: {ds_name}")
            except Exception as e:
                logger.error(f"✗ Failed to process {ds_name}: {e}")
                self.checkpoint.mark_failed(ds_name, str(e))

        # Generate final report
        self.generate_report()

        logger.info("\n" + "=" * 60)
        logger.info("Processing complete!")
        logger.info("=" * 60)

    def process_dataset(self, ds_conf: Dict, skip_processing: bool = False):
        """Process a single dataset with robust error handling."""
        lang = ds_conf.get('language', 'unknown')
        ds_name = ds_conf.get('name')

        # Determine output path
        output_path = os.path.join(self.output_dir, lang, ds_name)

        # Check if already processed
        if os.path.exists(output_path):
            logger.info(f"Dataset already exists at {output_path}, skipping download")
            try:
                ds = load_from_disk(output_path)
                ds_conf['_original_size'] = len(ds)
                ds_conf['_final_size'] = len(ds)
                return
            except:
                logger.warning(f"Could not load existing dataset, will reprocess")
                shutil.rmtree(output_path, ignore_errors=True)

        # Create language directory
        os.makedirs(os.path.join(self.output_dir, lang), exist_ok=True)

        # Load dataset with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                dataset = self._load_dataset(ds_conf)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Load attempt {attempt+1} failed: {e}. Retrying in 10s...")
                    time.sleep(10)
                else:
                    raise

        original_len = len(dataset)
        ds_conf['_original_size'] = original_len
        logger.info(f"Loaded {original_len} examples from {ds_name}")

        if skip_processing:
            # RAW COPY MODE - Just rename columns
            dataset = self._process_raw_copy(dataset, ds_conf)
        else:
            # NORMAL PROCESSING MODE
            dataset = self._process_normal(dataset, ds_conf)

        final_len = len(dataset)
        ds_conf['_final_size'] = final_len

        # Save locally
        logger.info(f"Saving to {output_path}")
        dataset.save_to_disk(output_path)

        # Push to Hub if configured
        if self.output_org and self.hf_token:
            self._push_to_hub(dataset, lang, ds_name)

    def _load_dataset(self, ds_conf: Dict) -> Dataset:
        """Load dataset from HuggingFace with proper configuration."""
        split = ds_conf.get('split', 'train')
        hf_path = ds_conf.get('hf_path')
        subset = ds_conf.get('subset')

        logger.info(f"Loading {hf_path}" + (f" ({subset})" if subset else ""))

        # Build load kwargs
        load_kwargs = {
            'split': split,
            'token': self.hf_token
        }

        if subset:
            load_kwargs['name'] = subset

        dataset = load_dataset(hf_path, **load_kwargs)

        # Handle Audio column to prevent auto-decoding
        voice_col = ds_conf.get('voice_col')
        if voice_col and voice_col in dataset.column_names:
            try:
                dataset = dataset.cast_column(voice_col, Value("string"))
            except:
                pass

        return dataset

    def _process_raw_copy(self, dataset: Dataset, ds_conf: Dict) -> Dataset:
        """Process dataset in raw copy mode (no filtering/resampling)."""
        logger.info("Running in RAW COPY mode (no processing)")

        voice_col = ds_conf.get('voice_col')
        text_col = ds_conf.get('text_col')

        # Validate columns exist
        if text_col not in dataset.column_names:
            raise ValueError(f"Text column '{text_col}' not found. Available: {dataset.column_names}")
        if voice_col not in dataset.column_names:
            raise ValueError(f"Voice column '{voice_col}' not found. Available: {dataset.column_names}")

        # Rename columns
        if voice_col != 'audio':
            if 'audio' in dataset.column_names:
                dataset = dataset.remove_columns(['audio'])
            dataset = dataset.rename_column(voice_col, 'audio')

        if text_col != 'text':
            if 'text' in dataset.column_names:
                dataset = dataset.remove_columns(['text'])
            dataset = dataset.rename_column(text_col, 'text')

        # Keep only necessary columns
        keep_cols = ['audio', 'text']
        if 'source' in dataset.column_names:
            keep_cols.append('source')
        dataset = dataset.select_columns([c for c in keep_cols if c in dataset.column_names])

        return dataset

    def _process_normal(self, dataset: Dataset, ds_conf: Dict) -> Dataset:
        """Process dataset with filtering and transformations."""
        logger.info("Running in NORMAL processing mode")

        voice_col = ds_conf.get('voice_col')
        text_col = ds_conf.get('text_col')

        # Filter function
        def filter_fn(example):
            # Text filtering
            if text_col and text_col in example:
                clean_text = self.text_proc.basic_clean(example[text_col])
                if not self.text_proc.filter_rules(clean_text):
                    return False

            # Audio duration filtering
            if ds_conf.get('type') == 'voice' and voice_col:
                if voice_col in example:
                    val = example[voice_col]
                    audio_path = val if isinstance(val, str) else val.get('path')

                    if not audio_path:
                        return False

                    duration = self.audio_proc.get_duration(audio_path)
                    if not self.audio_proc.is_valid_duration(duration):
                        return False

            return True

        logger.info("Applying filters...")
        dataset = dataset.filter(filter_fn)

        # Map function
        def map_fn(example):
            # Process text
            if text_col and text_col in example:
                example['text'] = self.text_proc.basic_clean(example[text_col])
                if text_col != 'text':
                    del example[text_col]

            # Process audio
            if ds_conf.get('type') == 'voice' and voice_col:
                if voice_col in example:
                    val = example[voice_col]
                    audio_path = val if isinstance(val, str) else val.get('path')
                    y, sr = self.audio_proc.load_audio(audio_path)

                    if y is not None:
                        example['audio'] = {
                            'array': y,
                            'sampling_rate': sr
                        }
                        if voice_col != 'audio':
                            del example[voice_col]

            return example

        logger.info("Processing audio and text...")
        dataset = dataset.map(map_fn)

        return dataset

    def _push_to_hub(self, dataset: Dataset, lang: str, ds_name: str):
        """Push dataset to HuggingFace Hub with retry logic."""
        if not self.output_org or not self.hf_token:
            return

        repo_id = f"{self.output_org}/african_speech_{lang}"

        logger.info(f"Pushing to Hub: {repo_id}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                dataset.push_to_hub(
                    repo_id,
                    token=self.hf_token,
                    commit_message=f"Add {ds_name} data"
                )
                logger.info(f"✓ Successfully pushed {ds_name} to {repo_id}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Push attempt {attempt+1} failed: {e}. Retrying in 10s...")
                    time.sleep(10)
                else:
                    logger.error(f"Failed to push to Hub: {e}")

    def merge_and_push(self, languages: Optional[List[str]] = None):
        """
        Merge processed datasets by language and push to Hub.

        Creates proper splits for each language code.
        """
        logger.info("\n" + "=" * 60)
        logger.info("MERGING AND PUSHING BY LANGUAGE")
        logger.info("=" * 60)

        if not os.path.exists(self.output_dir):
            logger.warning("No processed data found to merge.")
            return

        # Determine which languages to process
        if languages:
            lang_dirs = [l for l in os.listdir(self.output_dir)
                        if os.path.isdir(os.path.join(self.output_dir, l)) and l in languages]
        else:
            lang_dirs = [l for l in os.listdir(self.output_dir)
                        if os.path.isdir(os.path.join(self.output_dir, l))]

        for lang in lang_dirs:
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing Language: {lang}")
            logger.info(f"{'='*40}")

            lang_path = os.path.join(self.output_dir, lang)
            lang_datasets = []

            # Load all datasets for this language
            for ds_name in os.listdir(lang_path):
                ds_path = os.path.join(lang_path, ds_name)
                if not os.path.isdir(ds_path):
                    continue

                try:
                    logger.info(f"  Loading {ds_name}...")
                    ds = load_from_disk(ds_path)

                    # Add source metadata
                    if 'source' not in ds.column_names:
                        ds = ds.add_column("source", [ds_name] * len(ds))

                    # Keep common columns
                    keep_cols = ['audio', 'text', 'source']
                    ds = ds.select_columns([c for c in keep_cols if c in ds.column_names])

                    # Cast audio
                    if 'audio' in ds.column_names:
                        ds = ds.cast_column('audio', Audio(sampling_rate=16000))

                    lang_datasets.append(ds)
                    logger.info(f"    Loaded {len(ds)} examples")

                except Exception as e:
                    logger.error(f"  Failed to load {ds_name}: {e}")

            if lang_datasets:
                try:
                    # Concatenate all datasets for this language
                    merged_ds = concatenate_datasets(lang_datasets)
                    logger.info(f"  Merged total: {len(merged_ds)} examples")

                    # Push to Hub
                    if self.output_org and self.hf_token:
                        repo_id = f"{self.output_org}/african_speech_corpus"
                        config_name = lang

                        logger.info(f"  Pushing to {repo_id} (config: {config_name})...")

                        merged_ds.push_to_hub(
                            repo_id,
                            config_name=config_name,
                            token=self.hf_token,
                            commit_message=f"Merged {lang} data"
                        )

                        logger.info(f"  ✓ Successfully pushed {lang}")

                        # Update checkpoint
                        self.checkpoint.state.languages_completed.append(lang)
                        self.checkpoint.save()

                except Exception as e:
                    logger.error(f"  Failed to merge/push {lang}: {e}")
            else:
                logger.warning(f"  No valid datasets found for {lang}")

    def generate_report(self):
        """Generate processing report."""
        logger.info("\n" + "=" * 60)
        logger.info("FINAL REPORT")
        logger.info("=" * 60)

        data = []
        total_original = 0
        total_final = 0

        for ds_name, info in self.checkpoint.state.datasets_processed.items():
            original = info.get('original_size', 0)
            final = info.get('final_size', 0)
            status = info.get('status', 'unknown')

            total_original += original
            total_final += final

            data.append({
                "Dataset": ds_name,
                "Language": info.get('language', 'unknown'),
                "Original": original,
                "Final": final,
                "Retained": f"{(final/original)*100:.1f}%" if original > 0 else "0%",
                "Status": status
            })

        df = pd.DataFrame(data)
        print("\n" + df.to_markdown(index=False))

        # Save report
        df.to_csv("processing_report.csv", index=False)

        # Summary
        print(f"\nTotal Original: {total_original}")
        print(f"Total Final: {total_final}")
        print(f"Overall Retention: {(total_final/total_original)*100:.1f}%" if total_original > 0 else "0%")

        # Save checkpoint state
        self.checkpoint.save()


def reset_checkpoint(checkpoint_dir: str = "./checkpoints"):
    """Reset all checkpoints to allow reprocessing."""
    state_file = os.path.join(checkpoint_dir, "pipeline_state.json")
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"Checkpoint reset: {state_file}")
    else:
        print("No checkpoint found to reset")


def main():
    parser = argparse.ArgumentParser(
        description="African Speech Data Pipeline - Robust Edition"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--lang",
        help="Filter datasets by language code (e.g., 'amh', 'som', 'hau')"
    )
    parser.add_argument(
        "--dataset",
        help="Process only this specific dataset name"
    )
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="Skip processing and only merge existing datasets"
    )
    parser.add_argument(
        "--skip_processing",
        action="store_true",
        help="Raw copy mode - no filtering/resampling"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="./checkpoints",
        help="Directory for checkpoint files"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all checkpoints to reprocess everything"
    )
    parser.add_argument(
        "--push_only",
        action="store_true",
        help="Only push to Hub without processing"
    )

    args = parser.parse_args()

    # Reset checkpoints if requested
    if args.reset:
        reset_checkpoint(args.checkpoint_dir)

    # Check config exists
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    # Create pipeline
    pipeline = DataPipeline(args.config, args.checkpoint_dir)

    # Determine which mode to run
    if args.merge_only:
        # Only merge existing data
        languages = [args.lang] if args.lang else None
        pipeline.merge_and_push(languages)
    elif args.push_only:
        # Only push to Hub
        languages = [args.lang] if args.lang else None
        pipeline.merge_and_push(languages)
    else:
        # Run processing
        pipeline.run(
            lang_filter=args.lang,
            skip_processing=args.skip_processing,
            dataset_name=args.dataset
        )

        # Merge after processing
        languages = [args.lang] if args.lang else None
        pipeline.merge_and_push(languages)


if __name__ == "__main__":
    main()
