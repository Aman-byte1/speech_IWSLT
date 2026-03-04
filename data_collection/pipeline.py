#!/usr/bin/env python3
"""
Unified African Speech Data Pipeline
=====================================
Handles BOTH:
  - HuggingFace datasets  (source: hf)
  - Mozilla Common Voice 24.0 via DataCollective API  (source: mozilla_cv)

Usage:
  python pipeline.py --config config/amh.yaml
  python pipeline.py --config config/amh.yaml --lang amh
  python pipeline.py --config config/amh.yaml --merge-only
  python pipeline.py --config config/amh.yaml --status
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["HF_HUB_DISABLE_XET"] = "1"

import yaml
import pandas as pd
import numpy as np
import requests
import soundfile as sf
from mutagen import File as MutagenFile
from tqdm import tqdm

from datasets import (
    Audio,
    Dataset,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from datasets.features import Sequence

LOG_FMT = "%(asctime)s │ %(levelname)-7s │ %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("pipeline")

# ─────────────────── Mozilla API ───────────────────
MOZ_API_BASE = "https://datacollective.mozillafoundation.org/api"
SPLITS_TO_TRY = ["train.tsv", "dev.tsv", "test.tsv", "validated.tsv"]


# ─────────────────── Checkpoint ───────────────────
class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"datasets": {}, "pushed": {}}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def ds_done(self, name: str) -> bool:
        return self.data["datasets"].get(name, {}).get("status") == "done"

    def set_ds_done(self, name: str, rows: int, out_dir: str):
        self.data["datasets"][name] = {
            "status": "done",
            "rows": int(rows),
            "out_dir": out_dir,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()

    def set_ds_failed(self, name: str, err: str):
        self.data["datasets"][name] = {
            "status": "failed",
            "error": str(err)[:800],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()

    def lang_done(self, lang: str) -> bool:
        return self.data["pushed"].get(lang, {}).get("status") == "done"

    def set_lang_done(self, lang: str, rows: int):
        self.data["pushed"][lang] = {
            "status": "done",
            "rows": int(rows),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()


# ─────────────────── Text cleaning ───────────────────
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def clean_text(s: Any, lowercase: bool = True) -> str:
    if not isinstance(s, str):
        return ""
    s = _TAG_RE.sub("", s)
    s = s.replace("\u200b", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s.lower() if lowercase else s


# ─────────────────── Duration helpers ───────────────────
def _duration_from_path(path: str) -> Optional[float]:
    if not path:
        return None
    ext = Path(path).suffix.lower()
    try:
        if ext in {".mp3", ".ogg", ".opus", ".flac", ".wav", ".m4a", ".aac"}:
            m = MutagenFile(path)
            if m is not None and getattr(m, "info", None) is not None:
                length = getattr(m.info, "length", None)
                if length is not None:
                    return float(length)
    except Exception:
        pass
    try:
        info = sf.info(path)
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None

def _duration_from_bytes(b: bytes) -> Optional[float]:
    if not b:
        return None
    try:
        with sf.SoundFile(io.BytesIO(b)) as f:
            if f.frames and f.samplerate:
                return float(f.frames) / float(f.samplerate)
    except Exception:
        pass
    try:
        m = MutagenFile(fileobj=io.BytesIO(b))
        if m is not None and getattr(m, "info", None) is not None:
            length = getattr(m.info, "length", None)
            if length is not None:
                return float(length)
    except Exception:
        pass
    return None

def duration_from_audio_field(a: Any, default_sr: int) -> Optional[float]:
    if isinstance(a, list) and a:
        for item in a:
            d = duration_from_audio_field(item, default_sr)
            if d is not None:
                return d
        return None
    if a is None:
        return None
    if isinstance(a, str):
        return _duration_from_path(a)
    if isinstance(a, dict):
        if a.get("path"):
            d = _duration_from_path(a["path"])
            if d is not None:
                return d
        if a.get("bytes"):
            d = _duration_from_bytes(a["bytes"])
            if d is not None:
                return d
        if a.get("array") is not None:
            arr = a["array"]
            sr = int(a.get("sampling_rate") or default_sr)
            try:
                return float(len(arr)) / float(sr) if sr > 0 else None
            except Exception:
                return None
        return None
    if isinstance(a, (tuple, np.ndarray)):
        return float(len(a)) / float(default_sr) if default_sr > 0 else None
    return None


# ─────────────────── Mozilla CV helpers ───────────────────

def _moz_get_download_url(dataset_id: str, moz_api_key: str, retries: int = 3) -> str:
    """Call Mozilla DataCollective API to get a signed download URL."""
    url = f"{MOZ_API_BASE}/datasets/{dataset_id}/download"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {moz_api_key}", "Content-Type": "application/json"},
                timeout=120,
            )
            r.raise_for_status()
            j = r.json()
            dl = j.get("downloadUrl")
            if not dl:
                raise RuntimeError(f"No downloadUrl in response: keys={list(j.keys())}")
            return dl
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(10 * attempt)
    raise RuntimeError(f"Failed to get download URL after {retries} attempts: {last_err}")


def _download_file(url: str, out_path: Path):
    """Download a file with progress bar. Skips if already downloaded."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info(f"  Already downloaded: {out_path.name}")
        return

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0") or 0)
        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {out_path.name}"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _extract_tar(tar_path: Path, out_dir: Path):
    """Extract tar.gz archive. Skips if already extracted."""
    if out_dir.exists() and any(out_dir.iterdir()):
        log.info(f"  Already extracted: {out_dir.name}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  Extracting {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)


def _find_cv_root(extract_dir: Path, cv_folder: str) -> Path:
    """Locate the Common Voice root containing clips/ and split TSVs."""
    candidates = [extract_dir / cv_folder, extract_dir]
    for c in candidates:
        if (c / "clips").exists() and any((c / s).exists() for s in SPLITS_TO_TRY):
            return c
    for clips in extract_dir.rglob("clips"):
        parent = clips.parent
        if any((parent / s).exists() for s in SPLITS_TO_TRY):
            return parent
    raise FileNotFoundError(f"Could not locate Common Voice structure under {extract_dir}")


def _load_cv_tsvs(cv_root: Path) -> pd.DataFrame:
    """Load and merge all available split TSVs (train, dev, test, validated)."""
    dfs = []
    for split in SPLITS_TO_TRY:
        p = cv_root / split
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t", quoting=3, on_bad_lines="skip")
        if "path" in df.columns and "sentence" in df.columns:
            dfs.append(df[["path", "sentence"]])
    if not dfs:
        raise FileNotFoundError(f"No usable TSV splits found in {cv_root}")
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["path"])


# ─────────────────── Pipeline ───────────────────
class Pipeline:
    def __init__(self, cfg_path: str):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))

        self.data_dir = Path(self.cfg.get("data_dir", "./processed_data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt = Checkpoint(self.data_dir / "checkpoint.json")

        self.repo_id = self.cfg.get("repo_id")
        self.hf_token = self.cfg.get("hf_token") or os.environ.get("HF_TOKEN")
        self.moz_api_key = self.cfg.get("moz_api_key") or os.environ.get("MOZ_API_KEY")

        # Mozilla CV workspace directory
        self.moz_work_dir = Path(self.cfg.get("moz_work_dir", "./moz_cv_work"))

        audio_cfg = self.cfg.get("audio", {})
        self.target_sr = int(audio_cfg.get("sampling_rate", 16000))
        self.min_dur = float(audio_cfg.get("min_duration", 0.5))
        self.max_dur = float(audio_cfg.get("max_duration", 30.0))
        self.filter_by_duration = bool(audio_cfg.get("filter_by_duration", True))

        text_cfg = self.cfg.get("text", {})
        self.min_chars = int(text_cfg.get("min_chars", 2))
        self.max_chars = int(text_cfg.get("max_chars", 500))
        self.lowercase = bool(text_cfg.get("lowercase", True))

        self.num_proc = int(self.cfg.get("num_proc", 1))
        self.push_max_shard_size = self.cfg.get("push_max_shard_size", "500MB")

    def run(self, lang: Optional[str], merge_only: bool):
        if not merge_only:
            self.process_all(lang)
        self.merge_and_push(lang)
        self.report()

    def process_all(self, lang: Optional[str]):
        for dsc in self.cfg.get("datasets", []):
            if not dsc.get("enabled", True):
                continue

            name = dsc["name"]
            lc = dsc["language"]
            if lang and lc != lang:
                continue

            out_dir = self.data_dir / lc / name
            if self.ckpt.ds_done(name) and out_dir.exists():
                log.info(f"✔ SKIP {name} (already processed)")
                continue

            source = dsc.get("source", "hf")
            log.info(f"\n{'-'*72}\n▶ START {name} (lang={lc}, source={source})")

            try:
                # ── Branch by source type ──
                if source == "mozilla_cv":
                    ds = self._load_mozilla_cv(dsc)
                else:
                    ds = self._load_hf(dsc)

                log.info(f"  Loaded {len(ds)} rows. cols={ds.column_names}")

                # ── Normalize columns ──
                if source != "mozilla_cv":
                    # mozilla_cv already returns normalized columns
                    ds = self._normalize(ds, dsc)

                # 1) Make audio NON-decoding safely
                ds = self._force_audio_decode_false(ds)

                # 2) If audio is a list (Sequence(Audio)), unwrap to first element
                ds = self._unwrap_audio_if_list(ds)

                # 3) Filter and clean
                before = len(ds)
                ds = ds.filter(self._filter_fn, num_proc=self.num_proc, desc=f"Filter {name}")
                ds = ds.map(self._clean_map, num_proc=self.num_proc, desc=f"Clean {name}")
                after = len(ds)
                log.info(f"  Filtered: {before} → {after}")
                if after == 0:
                    raise RuntimeError("Empty after filtering")

                out_dir.mkdir(parents=True, exist_ok=True)
                ds.save_to_disk(str(out_dir))
                self.ckpt.set_ds_done(name, after, str(out_dir))
                log.info(f"✔ DONE {name} → {out_dir}")

            except Exception as e:
                log.error(f"✘ FAIL {name} → {e}")
                self.ckpt.set_ds_failed(name, str(e))

    # ─────────────────── HuggingFace loader ───────────────────

    def _load_hf(self, dsc: dict) -> Dataset:
        hf_path = dsc["hf_path"]
        subset = dsc.get("subset")
        split = dsc.get("split", "train")

        if "+" in split:
            parts = []
            for s in split.split("+"):
                try:
                    parts.append(load_dataset(hf_path, subset, split=s, token=self.hf_token))
                except Exception as e:
                    log.warning(f"  split '{s}' missing for {hf_path}/{subset}: {e}")
            if not parts:
                raise RuntimeError(f"No splits loaded for {hf_path} subset={subset}")
            return concatenate_datasets(parts)

        return load_dataset(hf_path, subset, split=split, token=self.hf_token)

    # ─────────────────── Mozilla Common Voice loader ───────────────────

    def _load_mozilla_cv(self, dsc: dict) -> Dataset:
        """
        Download, extract, parse TSVs, and build a Dataset from Mozilla
        Common Voice 24.0 archive using the DataCollective API.
        Returns a Dataset with columns: audio, text, lang, source.
        """
        if not self.moz_api_key:
            raise RuntimeError(
                "MOZ_API_KEY not set. Set it in config (moz_api_key) "
                "or env var (export MOZ_API_KEY=...)."
            )

        dataset_id = dsc.get("moz_dataset_id")
        if not dataset_id:
            raise RuntimeError(f"No moz_dataset_id in config for {dsc['name']}")

        cv_folder = dsc.get("cv_folder", "")
        lc = dsc["language"]
        name = dsc["name"]

        tar_path = self.moz_work_dir / f"cv24_{lc}.tar.gz"
        extract_dir = self.moz_work_dir / f"cv24_{lc}_extracted"

        # ── Download ──
        log.info(f"  [Mozilla CV] Getting download URL for {lc} ...")
        dl_url = _moz_get_download_url(dataset_id, self.moz_api_key)
        _download_file(dl_url, tar_path)

        # ── Extract ──
        _extract_tar(tar_path, extract_dir)

        # ── Find root and load TSVs ──
        cv_root = _find_cv_root(extract_dir, cv_folder)
        log.info(f"  [Mozilla CV] cv_root = {cv_root}")

        df = _load_cv_tsvs(cv_root)
        clips_dir = cv_root / "clips"

        # ── Build rows: filter + pack audio bytes ──
        audio_list: List[Dict] = []
        texts: List[str] = []

        for rel, sent in tqdm(
            zip(df["path"].astype(str), df["sentence"].astype(str)),
            total=len(df),
            desc=f"Build {name}",
        ):
            p = clips_dir / rel
            if not p.exists():
                continue

            t = sent.strip()
            if self.lowercase:
                t = t.lower()
            if not (self.min_chars <= len(t) <= self.max_chars):
                continue

            # Duration check (header-based, no waveform decode)
            d = _duration_from_path(str(p))
            if d is not None and not (self.min_dur <= d <= self.max_dur):
                continue

            # Pack audio bytes for portability on HF
            b = p.read_bytes()
            audio_list.append({"path": p.name, "bytes": b})
            texts.append(t)

        log.info(f"  [Mozilla CV] {lc}: {len(texts)} rows after filtering")

        if not texts:
            raise RuntimeError(f"Mozilla CV {lc}: 0 rows after filtering")

        ds = Dataset.from_dict({
            "audio": audio_list,
            "text": texts,
            "lang": [lc] * len(texts),
            "source": [name] * len(texts),
        })
        ds = ds.cast_column("audio", Audio(decode=False))
        return ds

    # ─────────────────── Column normalization ───────────────────

    def _normalize(self, ds: Dataset, dsc: dict) -> Dataset:
        audio_col = dsc.get("audio_col", dsc.get("voice_col", "audio"))
        text_col = dsc.get("text_col", "text")
        lc = dsc["language"]
        src = dsc["name"]

        if audio_col not in ds.column_names:
            raise KeyError(f"audio_col '{audio_col}' not in {ds.column_names}")
        if text_col not in ds.column_names:
            raise KeyError(f"text_col '{text_col}' not in {ds.column_names}")

        if audio_col != "audio":
            if "audio" in ds.column_names:
                ds = ds.remove_columns(["audio"])
            ds = ds.rename_column(audio_col, "audio")

        if text_col != "text":
            if "text" in ds.column_names:
                ds = ds.remove_columns(["text"])
            ds = ds.rename_column(text_col, "text")

        ds = ds.remove_columns([c for c in ds.column_names if c not in ("audio", "text")])
        ds = ds.add_column("lang", [lc] * len(ds))
        ds = ds.add_column("source", [src] * len(ds))
        return ds

    # ─────────────────── Audio feature helpers ───────────────────

    def _force_audio_decode_false(self, ds: Dataset) -> Dataset:
        try:
            return ds.cast_column("audio", Audio(sampling_rate=self.target_sr, decode=False))
        except Exception as e1:
            log.warning(f"  audio cast Audio(decode=False) failed: {e1}")
            try:
                return ds.cast_column("audio", Sequence(Audio(sampling_rate=self.target_sr, decode=False)))
            except Exception as e2:
                log.warning(f"  audio cast Sequence(Audio(decode=False)) failed: {e2}")
                return ds

    def _unwrap_audio_if_list(self, ds: Dataset) -> Dataset:
        feat = ds.features.get("audio")
        is_seq_audio = isinstance(feat, Sequence) and isinstance(feat.feature, Audio)
        if not is_seq_audio:
            return ds

        def pick_first(ex):
            a = ex["audio"]
            if isinstance(a, list) and len(a) > 0:
                return {"audio": a[0]}
            return {"audio": None}

        ds = ds.map(pick_first, num_proc=self.num_proc, desc="UnwrapAudio")
        ds = ds.filter(lambda ex: ex["audio"] is not None, num_proc=self.num_proc, desc="DropNullAudio")
        ds = ds.cast_column("audio", Audio(sampling_rate=self.target_sr, decode=False))
        return ds

    # ─────────────────── Filter / Clean ───────────────────

    def _filter_fn(self, ex: Dict[str, Any]) -> bool:
        t = clean_text(ex.get("text"), lowercase=self.lowercase)
        if not t:
            return False
        if not (self.min_chars <= len(t) <= self.max_chars):
            return False

        if not self.filter_by_duration:
            return True

        d = duration_from_audio_field(ex.get("audio"), default_sr=self.target_sr)
        if d is None:
            return True
        return self.min_dur <= d <= self.max_dur

    def _clean_map(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        ex["text"] = clean_text(ex.get("text"), lowercase=self.lowercase)
        return ex

    # ─────────────────── Merge & Push ───────────────────

    def merge_and_push(self, lang: Optional[str]):
        if not self.repo_id or not self.hf_token:
            log.warning("repo_id or HF token missing; skipping push.")
            return

        log.info(f"\n{'='*72}\nMERGE & PUSH\n{'='*72}")

        for lang_dir in sorted(self.data_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            lc = lang_dir.name
            if lc.startswith("."):
                continue
            if lang and lc != lang:
                continue
            if self.ckpt.lang_done(lc):
                log.info(f"✔ SKIP push {lc} (already pushed)")
                continue

            parts: List[Dataset] = []
            for ds_dir in sorted(lang_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                try:
                    d = load_from_disk(str(ds_dir))
                    keep = [c for c in ["audio", "text", "lang", "source"] if c in d.column_names]
                    drop = [c for c in d.column_names if c not in keep]
                    if drop:
                        d = d.remove_columns(drop)

                    d = d.cast_column("audio", Audio(sampling_rate=self.target_sr, decode=False))
                    parts.append(d)
                    log.info(f"  + {ds_dir.name:35s} {len(d):8d} rows")
                except Exception as e:
                    log.error(f"  - failed load {ds_dir.name}: {e}")

            if not parts:
                log.warning(f"No datasets for {lc}")
                continue

            merged = concatenate_datasets(parts)
            log.info(f"  ⇒ {lc} merged rows: {len(merged)}")

            for attempt in range(1, 4):
                try:
                    log.info(f"  Pushing {lc} (attempt {attempt}/3)")
                    merged.push_to_hub(
                        self.repo_id,
                        config_name=lc,
                        token=self.hf_token,
                        max_shard_size=self.push_max_shard_size,
                        commit_message=f"Add {lc} ({len(merged)} rows)",
                    )
                    self.ckpt.set_lang_done(lc, len(merged))
                    log.info(f"  ✔ pushed {lc}")
                    break
                except Exception as e:
                    log.error(f"  push failed: {e}")
                    if attempt < 3:
                        time.sleep(20 * attempt)

    # ─────────────────── Report ───────────────────

    def report(self):
        log.info(f"\n{'='*72}\nREPORT\n{'='*72}")
        rows = []
        for name, info in sorted(self.ckpt.data.get("datasets", {}).items()):
            rows.append({
                "Dataset": name,
                "Status": info.get("status", "?"),
                "Rows": info.get("rows", "–"),
                "Timestamp": info.get("ts", ""),
                "Error": (info.get("error") or "")[:80],
            })
        if rows:
            df = pd.DataFrame(rows)
            try:
                print("\n" + df.to_markdown(index=False))
            except Exception:
                print("\n" + df.to_string(index=False))
            out_csv = self.data_dir / "report.csv"
            df.to_csv(out_csv, index=False)
            log.info(f"CSV saved → {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Unified African Speech Data Pipeline")
    ap.add_argument("--config", default="config.yaml", help="Path to language YAML config")
    ap.add_argument("--lang", help="Process only this language code")
    ap.add_argument("--merge-only", action="store_true", help="Skip processing, only merge and push")
    ap.add_argument("--status", action="store_true", help="Show processing report only")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    pipe = Pipeline(args.config)

    if args.status:
        pipe.report()
        return

    pipe.run(lang=args.lang, merge_only=args.merge_only)


if __name__ == "__main__":
    main()