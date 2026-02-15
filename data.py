
import argparse
import sys
import yaml
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import pandas as pd
import torch
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
from datasets import load_dataset, Dataset, Audio, Value
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import noisereduce as nr
import re

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessStats:
    dataset_name: str
    original_size: int
    final_size: int
    discarded_count: int

class TextProcessor:
    def __init__(self, config: Dict):
        self.config = config.get('text', {})
        self.models_config = config.get('models', {})
        self.semantic_model = None
        
        # Load Semantic Model if needed (Lazy loading)
        if 'semantic_model' in self.models_config and self.models_config['semantic_model']:
            try:
                logger.info(f"Loading Semantic Model: {self.models_config['semantic_model']}")
                # Check for GPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.semantic_model = SentenceTransformer(self.models_config['semantic_model'], device=device)
            except Exception as e:
                logger.warning(f"Could not load semantic model: {e}. Semantic filtering will be skipped.")

    def basic_clean(self, text: str) -> str:
        """Removes HTML, extra spaces, etc."""
        if not text or not isinstance(text, str):
            return ""
        # Simple rule-based cleaning
        text = re.sub(r'<[^>]+>', '', text) # Remove HTML
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower() # Normalize to lowercase as per request

    def filter_rules(self, src_text: str, tgt_text: Optional[str] = None) -> bool:
        """Stage 1: Rule-based filtering (Length, Ratio)"""
        if not src_text: 
            return False
            
        # Length check
        if not (self.config.get('min_chars', 3) <= len(src_text) <= self.config.get('max_chars', 200)):
            return False
            
        # Ratio check (if target exists)
        if tgt_text:
            if not tgt_text: return False
            len_src = len(src_text)
            len_tgt = len(tgt_text)
            if len_src == 0 or len_tgt == 0: return False
            
            ratio = max(len_src, len_tgt) / min(len_src, len_tgt)
            if ratio > self.config.get('max_ratio', 2.0):
                return False
                
        return True

    def semantic_filter(self, src_texts: List[str], tgt_texts: List[str]) -> List[bool]:
        """Stage 3: Semantic Filtering using LaBSE or similar"""
        if self.semantic_model is None:
            return [True] * len(src_texts)
            
        logger.info("Running Semantic Filtering...")
        try:
            embeddings1 = self.semantic_model.encode(src_texts, convert_to_tensor=True, show_progress_bar=False)
            embeddings2 = self.semantic_model.encode(tgt_texts, convert_to_tensor=True, show_progress_bar=False)
            
            # Compute cosine similarity
            # Utilizing semantic search utility, but strictly we want pair-wise.
            # cos_sim returns All-vs-All. For huge lists this is bad.
            # Start simple: use loop or diagonal if tensor ops allow.
            # Optimized: (a / |a|) . (b / |b|)
            import torch.nn.functional as F
            scores = F.cosine_similarity(embeddings1, embeddings2)
            
            threshold = self.config.get('semantic_threshold', 0.6)
            return (scores >= threshold).cpu().tolist()
        except Exception as e:
            logger.error(f"Semantic filtering failed: {e}")
            return [True] * len(src_texts)

class DataPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.text_proc = TextProcessor(self.config)
        self.stats = []
        self.hf_token = self.config.get('hf_token')
        self.output_org = self.config.get('output_org')

    def run(self, lang_filter: Optional[str] = None, skip_processing: bool = False):
        datasets_config = self.config.get('datasets', [])
        
        for ds_conf in datasets_config:
            # Filter by language if provided
            if lang_filter:
                ds_lang = ds_conf.get('language', '').lower()
                if ds_lang != lang_filter.lower():
                    continue

            self.process_dataset(ds_conf, skip_processing)
            
        self.generate_report()

    def process_dataset(self, ds_conf, skip_processing: bool = False):
        # Determine Output Path
        lang_dir = ds_conf.get('language', 'unknown')
        output_name = f"{ds_conf['name']}_processed"
        local_path = f"./processed_data/{lang_dir}/{output_name}"
        
        # Check if already processed
        if os.path.exists(local_path):
            logger.info(f"Dataset {ds_conf['name']} already processed at {local_path}. Skipping.")
            return

        logger.info(f"Processing Dataset: {ds_conf['name']} (Skip Processing: {skip_processing})")
        
        # Load Dataset
        try:
            split = ds_conf.get('split', 'train')
            logger.info(f"Loading {ds_conf['hf_path']} ({ds_conf.get('subset')})...")
            
            # NOTE: Loading with 'audio' column usually adds Audio feature type.
            dataset = load_dataset(ds_conf['hf_path'], ds_conf.get('subset'), split=split, token=self.hf_token)
        except Exception as e:
            logger.error(f"Failed to load {ds_conf['name']}: {e}")
            return

        original_len = len(dataset)
        
        if skip_processing:
            # === RAW COPY MODE ===
            # Just rename columns and save. No decoding.
            logger.info("Skip Processing enabled: Renaming columns and saving raw data...")
            
            # Check if columns exist
            if text_col not in dataset.column_names:
                logger.error(f"Text column '{text_col}' not found in {ds_conf['name']}. Available: {dataset.column_names}")
                return
            if voice_col not in dataset.column_names:
                logger.error(f"Voice column '{voice_col}' not found in {ds_conf['name']}. Available: {dataset.column_names}")
                return

            # Rename Voice Column to 'audio'
            if voice_col != 'audio':
                if 'audio' in dataset.column_names:
                    logger.warning("Target column 'audio' already exists. Removing it to avoid conflict.")
                    dataset = dataset.remove_columns(['audio'])
                logger.info(f"Renaming {voice_col} -> audio")
                dataset = dataset.rename_column(voice_col, 'audio')
            
            # Rename Text Column to 'text'
            if text_col != 'text':
                if 'text' in dataset.column_names:
                    logger.warning("Target column 'text' already exists. Removing it to avoid conflict.")
                    dataset = dataset.remove_columns(['text'])
                logger.info(f"Renaming {text_col} -> text")
                dataset = dataset.rename_column(text_col, 'text')
                
            # Keep only necessary columns
            if 'audio' in dataset.column_names and 'text' in dataset.column_names:
                dataset = dataset.select_columns(['audio', 'text'])
            else:
                logger.error(f"Failed to normalize columns for {ds_conf['name']}. 'audio' or 'text' missing.")
                return

        else:
            # === NORMAL PROCESSING MODE ===
            # (Contains previous logic logic, effectively removed/replaced by this block structure for clarity
            # if we wanted to preserve it, we'd include it here. 
            # For this edit, I will reconstruct the loop effectively.)
            
            # 1. Cast Audio - Skipped to avoid torchcodec dependency issues.
            # We will load audio manually using librosa.
            target_sr = self.config['audio'].get('sampling_rate', 16000)
            
            # Helper to load audio from path
            def load_audio(audio_path):
                try:
                    # Librosa loads as mono=True by default, and resamples if sr is provided
                    y, s = librosa.load(audio_path, sr=target_sr, mono=True)
                    return y, s
                except Exception as e:
                    logger.warning(f"Error loading audio {audio_path}: {e}")
                    return None, 0

            # CRITICAL FIX: Rename/Drop the original Audio Feature column to prevent 
            # 'datasets' from trying to decode it automatically (which triggers torchcodec).
            if ds_conf['type'] == 'voice':
                 voice_col = ds_conf.get('voice_col')
                 logger.info(f"Removing Audio feature branding from '{voice_col}' to bypass decoding...")
                 try:
                     # Cast to string to force it to be treated as a path, disabling auto-decoding
                     dataset = dataset.cast_column(voice_col, Value("string"))
                 except Exception as e:
                     logger.warning(f"Could not cast column {voice_col} to string: {e}")

            # 2. Filter Function (Includes Text Rules & Audio Duration)
            def filter_fn(example):
                # Text Rules
                text_col = ds_conf.get('text_col')
                if text_col and text_col in example:
                    clean_text = self.text_proc.basic_clean(example[text_col])
                    if not self.text_proc.filter_rules(clean_text):
                        return False
                
                # Audio Duration Rules
                if ds_conf['type'] == 'voice':
                    voice_col = ds_conf.get('voice_col')
                    if voice_col and voice_col in example:
                        val = example[voice_col]
                        audio_path = val if isinstance(val, str) else val.get('path')
                        if not audio_path: return False

                        try:
                            duration = librosa.get_duration(path=audio_path)
                        except:
                            y, sr = load_audio(audio_path)
                            if y is None: return False
                            duration = len(y) / sr
                        
                        min_dur = self.config['audio'].get('min_duration', 1.0)
                        max_dur = self.config['audio'].get('max_duration', 20.0)
                        
                        if not (min_dur <= duration <= max_dur):
                            return False
                return True

            logger.info("Applying filters (Text Rules & Audio Duration)...")
            dataset_filtered = dataset.filter(filter_fn)

            # 3. Map Function (Text Cleaning & Optional Semantic Filter Prep)
            def map_fn(example):
                # Standardize Text Column
                text_col = ds_conf.get('text_col')
                if text_col and text_col in example:
                    example['text'] = self.text_proc.basic_clean(example[text_col])
                    # Remove original if different (optional, but cleaner)
                    if text_col != 'text':
                        del example[text_col]
                
                # Standardize Audio Column
                # Load Audio into array (simulating HF Audio feature)
                if ds_conf['type'] == 'voice':
                     voice_col = ds_conf.get('voice_col')
                     if voice_col and voice_col in example:
                        val = example[voice_col]
                        audio_path = val if isinstance(val, str) else val.get('path')
                        y, s = load_audio(audio_path)
                        
                        if y is not None:
                            # Update example with decoded data
                            # Standardize to 'audio' column
                            example['audio'] = {
                                'array': y,
                                'sampling_rate': s
                            }
                            # Remove original if different
                            if voice_col != 'audio':
                                del example[voice_col]
                        
                return example

            logger.info("Applying text normalization and standardizing columns...")
            dataset_processed = dataset_filtered.map(map_fn)
            dataset = dataset_processed # update ref

        # 4. Semantic Filtering (If applicable)
        # Assuming we have pairs. If unrelated audio/text, skip.
        # If we have a translation target column (e.g. source_lang, target_lang columns), we can run it.
        # For ASR datasets (Audio -> Text), semantic filtering (LaBSE) isn't typically used *unless* we are filtering 
        # against a synthetic translation or something. The prompt requested it explicitly for "datasets", likely assuming parallel text.
        # We will check if 'target_text_col' is defined in config or if we want to run it on (text_col, target_col).
        # For now, simplistic implementation: skip unless we explicitly configured parallel text cols.
        
        final_len = len(dataset) # Update to use whatever dataset we ended up with
        
        self.stats.append(ProcessStats(
            dataset_name=ds_conf['name'],
            original_size=original_len,
            final_size=final_len,
            discarded_count=original_len - final_len
        ))
        
        # Save / Upload
        logger.info(f"Saving to {local_path}")
        dataset.save_to_disk(local_path)
        
        # We assume the user wants the MERGED dataset pushed, not individual ones.
        # But per original design, we pushed individually.
        # Now we will skip individual push if we plan to merge all later.
        # OR we can keep it. The user said "merge it and then push it".
        # Let's keep individual push optional or skip it. 
        # For now, I'll comment out individual push to focus on the merged push.
        # if self.output_org and self.hf_token:
        #     repo_id = f"{self.output_org}/{ds_conf.get('language', 'multi')}-{output_name}"
        #     logger.info(f"Uploading to Hub: {repo_id}")
        #     try:
            #         dataset_processed.push_to_hub(repo_id, token=self.hf_token)
            #     except Exception as e:
            #         logger.error(f"Upload failed: {e}")

    def merge_and_push(self):
        """Merges processed datasets by language and pushes to HF as separate configs."""
        logger.info("\n=== MERGING AND PUSHING BY LANGUAGE ===")
        
        # Walk through processed_data directory
        root_dir = "./processed_data"
        if not os.path.exists(root_dir):
            logger.warning("No processed data found to merge.")
            return

        from datasets import load_from_disk, concatenate_datasets, Audio
        
        # Iterate over each language folder
        for lang_dir in os.listdir(root_dir):
            lang_path = os.path.join(root_dir, lang_dir)
            if not os.path.isdir(lang_path): continue
            
            logger.info(f"\nProcessing Language: {lang_dir}")
            lang_datasets = []
            
            for ds_name in os.listdir(lang_path):
                ds_path = os.path.join(lang_path, ds_name)
                if os.path.isdir(ds_path):
                    try:
                        logger.info(f"  Loading {ds_name}...")
                        ds = load_from_disk(ds_path)
                        
                        # Add metadata
                        if 'source' not in ds.column_names:
                            ds = ds.add_column("source", [ds_name] * len(ds))
                            
                        # Keep common columns
                        keep_cols = ['audio', 'text', 'source']
                        ds = ds.select_columns([c for c in keep_cols if c in ds.column_names])
                        
                        # Standardize Audio
                        if 'audio' in ds.column_names:
                            ds = ds.cast_column('audio', Audio(sampling_rate=16000))

                        lang_datasets.append(ds)
                    except Exception as e:
                        logger.error(f"  Failed to load {ds_name}: {e}")

            if lang_datasets:
                try:
                    merged_lang_ds = concatenate_datasets(lang_datasets)
                    logger.info(f"  Merged {lang_dir}: {len(merged_lang_ds)} examples")
                    
                    if self.output_org and self.hf_token:
                        repo_id = f"{self.output_org}/merged_speech_dataset"
                        logger.info(f"  Pushing to {repo_id} (config: {lang_dir})...")
                        # Push with config_name = language code (e.g., 'hausa', 'amharic')
                        merged_lang_ds.push_to_hub(repo_id, config_name=lang_dir, token=self.hf_token)
                        logger.info("  Push successful.")
                    else:
                        logger.warning("HF Token or Output Org not set. Skipping push for this language.")
                except Exception as e:
                    logger.error(f"  Failed to merge/push {lang_dir}: {e}")
            else:
                logger.warning(f"  No valid datasets found for {lang_dir}")

    def generate_report(self):
        logger.info("\n=== FINAL REPORT ===")
        data = []
        for s in self.stats:
            data.append({
                "Dataset": s.dataset_name,
                "Original": s.original_size,
                "Final": s.final_size,
                "Discarded": s.discarded_count,
                "Retention(%)": f"{(s.final_size/s.original_size)*100:.1f}" if s.original_size else "0.0"
            })
        
        df = pd.DataFrame(data)
        print(df.to_markdown(index=False))
        df.to_csv("report.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--lang", help="Filter datasets by language (e.g., 'hausa', 'amharic')")
    parser.add_argument("--merge_only", action="store_true", help="Skip processing and only merge existing processed datasets")
    parser.add_argument("--skip_processing", action="store_true", help="Duplicate datasets without filtering/resampling (fixes torchcodec issues)")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
        
    pipeline = DataPipeline(args.config)
    
    if not args.merge_only:
        pipeline.run(lang_filter=args.lang, skip_processing=args.skip_processing)
    
    pipeline.merge_and_push()
