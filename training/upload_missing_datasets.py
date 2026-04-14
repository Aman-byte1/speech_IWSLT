import os
import io
import argparse
import logging
from datasets import load_dataset, Audio, Dataset
import soundfile as sf
import librosa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Push to a "_clean" config so it cleanly overwrites past corrupted runs
CONFIGS = [
    {
        "lang": "French",
        "source_repo": "facebook/voxpopuli",
        "source_subset": "fr",
        "target_repo": "amanuelbyte/african_speech_dataset_fra",
        "target_subset": "fra_Latn_clean",  # New clean subset
    },
    {
        "lang": "Swahili",
        "source_repo": "amanuelbyte/african_speech_clean",
        "source_subset": "swahili",
        "target_repo": "amanuelbyte/african_speech_clean",
        "target_subset": "swahili_clean",
    },
    {
        "lang": "Yoruba",
        "source_repo": "amanuelbyte/african_speech_clean",
        "source_subset": "yoruba",
        "target_repo": "amanuelbyte/african_speech_clean",
        "target_subset": "yoruba_clean",
    },
    {
        "lang": "Spanish",
        "source_repo": "amanuelbyte/african_speech_dataset_spa",
        "source_subset": "spa_Latn",
        "target_repo": "amanuelbyte/african_speech_dataset_spa",
        "target_subset": "spa_Latn_clean",
    },
    {
        "lang": "Egyptian Arabic",
        "source_repo": "amanuelbyte/african_speech_dataset_new_uncleaned",
        "source_subset": "arz",
        "target_repo": "amanuelbyte/african_speech_dataset_new_uncleaned",
        "target_subset": "arz_clean",
    }
]

def format_row(ex):
    """Normalize text column."""
    text = ""
    if "normalized_text" in ex:
        text = ex["normalized_text"]
    elif "sentence" in ex:
        text = ex["sentence"]
    elif "transcription" in ex:
        text = ex["transcription"]
    elif "text" in ex:
        text = ex["text"]
    
    return {"audio": ex["audio"], "text": text}


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN missing. Please `export HF_TOKEN=...` before running.")
        return

    for cfg in CONFIGS:
        log.info(f"\n=============================================")
        log.info(f"🚀 Filtering 10k perfect samples for {cfg['lang']} ...")
        log.info(f"   Source: {cfg['source_repo']} ({cfg['source_subset']})")
        
        try:
            # We don't auto-decode audio here so we can parse it safely inside the generator
            ds_stream = load_dataset(
                cfg['source_repo'], 
                cfg['source_subset'], 
                split="train", 
                streaming=True, 
                token=token
            ).cast_column("audio", Audio(decode=False))

            def robust_generator():
                iterable = iter(ds_stream)
                valid_count = 0
                max_samples = 10000

                while valid_count < max_samples:
                    try:
                        ex = next(iterable)
                        
                        # Normalize text column
                        row = format_row(ex)
                        text = row["text"]
                        if not text or len(str(text).strip()) < 2:
                            continue
                            
                        # Safely verify audio integrity
                        audio_dict = row["audio"]
                        if not audio_dict:
                            continue
                            
                        audio_b = audio_dict.get("bytes")
                        if not audio_b and "path" in audio_dict:
                            with open(audio_dict["path"], "rb") as f:
                                audio_b = f.read()
                                
                        if not audio_b:
                            continue
                            
                        # Try decoding the audio. This natively drops the corrupted Spanish/Yoruba samples!
                        try:
                            with io.BytesIO(audio_b) as f:
                                array, sr = sf.read(f, dtype="float32")
                        except Exception:
                            # Fallback to librosa if soundfile fails
                            with io.BytesIO(audio_b) as f:
                                array, sr = librosa.load(f, sr=None, dtype="float32")
                                
                        input_length = len(array)
                                
                        # Apply PyTorch CTC crash-prevention filter
                        if (input_length // 320) <= len(text):
                            continue # Skip files where audio is too short for text
                            
                        # If we survived all integrity checks, yield it!
                        # The audio dict keeps the raw byte array so we don't balloon memory
                        yield row
                        valid_count += 1
                        
                        if valid_count % 500 == 0:
                            log.info(f"   ✓ Extracted {valid_count} / {max_samples} clean samples...")
                            
                    except StopIteration:
                        log.info("Reached end of source dataset.")
                        break
                    except Exception as e:
                        # Silently bypass any huggingface corrupted parquet block errors
                        pass

            # Materialize the verified samples
            log.info("Starting safe stream extraction...")
            clean_ds = Dataset.from_generator(robust_generator)
            
            # Formally apply the 16000Hz Audio feature so HF hub recognizes it as standard Audio
            clean_ds = clean_ds.cast_column("audio", Audio(sampling_rate=16000))
            
            log.info(f"Successfully vetted {len(clean_ds)} samples.")

            # Push securely to hub
            log.info(f"Pushing to {cfg['target_repo']} (subset: {cfg['target_subset']})")
            clean_ds.push_to_hub(
                repo_id=cfg['target_repo'],
                config_name=cfg['target_subset'], 
                split="train",
                token=token,
                private=False
            )
            log.info(f"✅ {cfg['lang']} extraction and upload complete!")

        except Exception as e:
            log.error(f"❌ Master process failed for {cfg['lang']}: {e}")

if __name__ == "__main__":
    main()
