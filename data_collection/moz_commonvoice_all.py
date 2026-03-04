#!/usr/bin/env python3
"""
Mozilla Common Voice (DataCollective API) -> Hugging Face unified pipeline (NO YAML)

- Downloads per-language tar.gz using Mozilla API
- Extracts, reads TSVs (train/dev/test/validated if present)
- Filters by text length + duration (header-based, no torchcodec)
- Stores audio as bytes (portable), Audio(decode=False)
- Saves to disk with checkpoint resume
- Pushes to ONE HF dataset repo with config_name per language code
- Optional backup push to another repo if the primary push fails

Run:
  export MOZ_API_KEY="..."
  export HF_TOKEN="hf_..."
  python moz_cv24_unified.py --repo_id amanuelbyte/african_speech_dataset_new_uncleaned

Optional backup:
  python moz_cv24_unified.py --repo_id amanuelbyte/african_speech_dataset_new_uncleaned \
      --backup_repo_id amanuelbyte/african_speech_dataset_new_uncleaned_backup
"""

import argparse
import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import pandas as pd
import requests
from tqdm import tqdm
from mutagen import File as MutagenFile
import soundfile as sf

from datasets import Dataset, Audio


API_BASE = "https://datacollective.mozillafoundation.org/api"

# ---------- DATASET IDs YOU PROVIDED (CV Scripted Speech 24.0) ----------
# Add som/wol when you have them.
DATASET_ID_BY_LANG: Dict[str, Optional[str]] = {
    "amh": "cmj8u3ort000lnxxbp7q3icgp",
    "som": None,  # TODO: add when you have it
    "hau": "cmj8u3p6w00alnxxbby9yqbms",
    "swh": "cmj8u3puq00qhnxxbg26y0owu",
    "yor": "cmj8u3q2500v5nxxb6xfa6jn5",
    "zul": "cmj8u3q3b00vxnxxbilr9nfyu",
    "wol": None,  # TODO: add when you have it
}

# Folder code inside Common Voice archives
CV_FOLDER_BY_LANG = {
    "amh": "am",
    "som": "so",
    "hau": "ha",
    "swh": "sw",
    "yor": "yo",
    "zul": "zu",
    "wol": "wo",
}

SPLITS_TO_TRY = ["train.tsv", "dev.tsv", "test.tsv", "validated.tsv"]


# ---------------- Checkpoint ----------------
class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"downloaded": {}, "extracted": {}, "built": {}, "pushed": {}}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def is_done(self, section: str, key: str) -> bool:
        return bool(self.data.get(section, {}).get(key))

    def set_done(self, section: str, key: str, value=True):
        self.data.setdefault(section, {})[key] = value
        self.save()


# ---------------- Mozilla API ----------------
def moz_get_download_url(dataset_id: str, moz_api_key: str, retries: int = 3) -> str:
    url = f"{API_BASE}/datasets/{dataset_id}/download"
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


def download_file(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
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


def extract_tar(tar_path: Path, out_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)


# ---------------- Common Voice parsing ----------------
def find_cv_root(extract_dir: Path, lang_code: str) -> Path:
    """
    Find folder containing:
      - clips/
      - at least one of SPLITS_TO_TRY
    Handles nested top-level folders.
    """
    cv_folder = CV_FOLDER_BY_LANG[lang_code]

    # direct common cases
    candidates = [extract_dir / cv_folder, extract_dir]
    for c in candidates:
        if (c / "clips").exists() and any((c / s).exists() for s in SPLITS_TO_TRY):
            return c

    # brute-force
    for clips in extract_dir.rglob("clips"):
        parent = clips.parent
        if any((parent / s).exists() for s in SPLITS_TO_TRY):
            return parent

    raise FileNotFoundError(f"Could not locate Common Voice structure under {extract_dir}")


def load_tsvs(cv_root: Path) -> pd.DataFrame:
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
    out = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["path"])
    return out


def duration_from_path(path: str) -> Optional[float]:
    """Fast duration from headers (no waveform decode)."""
    try:
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
        pass
    return None


def build_dataset(
    cv_root: Path,
    lang_code: str,
    min_chars: int,
    max_chars: int,
    min_dur: float,
    max_dur: float,
    lowercase: bool,
) -> Dataset:
    df = load_tsvs(cv_root)
    clips_dir = cv_root / "clips"

    # pre-filter rows before loading bytes
    audio_paths: List[str] = []
    texts: List[str] = []
    for rel, sent in zip(df["path"].astype(str), df["sentence"].astype(str)):
        p = clips_dir / rel
        if not p.exists():
            continue
        t = sent.strip()
        if lowercase:
            t = t.lower()
        if not (min_chars <= len(t) <= max_chars):
            continue

        d = duration_from_path(str(p))
        if d is not None and not (min_dur <= d <= max_dur):
            continue

        audio_paths.append(str(p))
        texts.append(t)

    ds = Dataset.from_dict(
        {
            "audio_path": audio_paths,
            "text": texts,
            "lang": [lang_code] * len(texts),
            "source": ["common_voice_mozilla_24.0"] * len(texts),
        }
    )

    # Pack bytes so it is portable on HF (no separate clips upload needed)
    def pack_batch(batch):
        audios = []
        for p in batch["audio_path"]:
            b = Path(p).read_bytes()
            audios.append({"path": Path(p).name, "bytes": b})
        return {"audio": audios}

    ds = ds.map(
        pack_batch,
        batched=True,
        batch_size=16,
        remove_columns=["audio_path"],
        desc=f"PackBytes {lang_code}",
    )

    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def push_with_fallback(
    ds: Dataset,
    lang_code: str,
    repo_id: str,
    hf_token: str,
    backup_repo_id: Optional[str],
    max_shard_size: str = "500MB",
) -> Tuple[bool, str]:
    """
    Try pushing to primary repo; if fails, optionally push to backup repo.
    Returns (success, repo_used).
    """
    def _push(target_repo: str) -> None:
        ds.push_to_hub(
            target_repo,
            config_name=lang_code,
            token=hf_token,
            max_shard_size=max_shard_size,
            commit_message=f"Mozilla CV 24.0 {lang_code} rows={len(ds)}",
        )

    # primary retries
    for attempt in range(1, 4):
        try:
            print(f"Push {lang_code} -> {repo_id} (attempt {attempt}/3)")
            _push(repo_id)
            return True, repo_id
        except Exception as e:
            print(f"Primary push failed: {e}")
            if attempt < 3:
                time.sleep(20 * attempt)

    if backup_repo_id:
        for attempt in range(1, 4):
            try:
                print(f"Backup push {lang_code} -> {backup_repo_id} (attempt {attempt}/3)")
                _push(backup_repo_id)
                return True, backup_repo_id
            except Exception as e:
                print(f"Backup push failed: {e}")
                if attempt < 3:
                    time.sleep(20 * attempt)

    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="HF dataset repo (e.g. amanuelbyte/african_speech_dataset_new_uncleaned)")
    ap.add_argument("--backup_repo_id", default=None, help="Optional backup HF repo if primary push fails")
    ap.add_argument("--work_dir", default="./moz_cv_work", help="Download/extract workspace")
    ap.add_argument("--out_dir", default="./processed_data_mozilla_cv24", help="Local processed datasets")
    ap.add_argument("--langs", default="amh,som,hau,swh,yor,zul,wol")
    ap.add_argument("--min_chars", type=int, default=2)
    ap.add_argument("--max_chars", type=int, default=500)
    ap.add_argument("--min_dur", type=float, default=0.5)
    ap.add_argument("--max_dur", type=float, default=30.0)
    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--no_lowercase", action="store_false", dest="lowercase")
    ap.add_argument("--max_shard_size", default="500MB")
    ap.add_argument("--no_push", action="store_true", help="Build/save locally but do not push")
    args = ap.parse_args()

    moz_api_key = os.environ.get("MOZ_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    if not moz_api_key:
        raise SystemExit("Missing MOZ_API_KEY env var")
    if not hf_token:
        raise SystemExit("Missing HF_TOKEN env var")

    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(out_dir / "checkpoint.json")

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    for lc in langs:
        dataset_id = DATASET_ID_BY_LANG.get(lc)
        if not dataset_id:
            print(f"[SKIP] {lc}: missing Mozilla dataset id (add it to DATASET_ID_BY_LANG)")
            continue

        tar_path = work_dir / f"cv24_{lc}.tar.gz"
        extract_dir = work_dir / f"cv24_{lc}_extracted"
        save_dir = out_dir / lc / "common_voice_mozilla_24.0"

        # ---- Download
        if not ckpt.is_done("downloaded", lc):
            dl_url = moz_get_download_url(dataset_id, moz_api_key)
            download_file(dl_url, tar_path)
            ckpt.set_done("downloaded", lc, True)
        else:
            print(f"[OK] {lc}: download already done")

        # ---- Extract
        if not ckpt.is_done("extracted", lc):
            extract_tar(tar_path, extract_dir)
            ckpt.set_done("extracted", lc, True)
        else:
            print(f"[OK] {lc}: extract already done")

        # ---- Build dataset
        if not ckpt.is_done("built", lc):
            cv_root = find_cv_root(extract_dir, lc)
            print(f"[INFO] {lc}: cv_root={cv_root}")

            ds = build_dataset(
                cv_root=cv_root,
                lang_code=lc,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                min_dur=args.min_dur,
                max_dur=args.max_dur,
                lowercase=args.lowercase,
            )
            print(f"[INFO] {lc}: rows={len(ds)}")

            save_dir.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(save_dir))
            ckpt.set_done("built", lc, True)
        else:
            print(f"[OK] {lc}: build already done")
            ds = Dataset.load_from_disk(str(save_dir))  # type: ignore

        # ---- Push
        if args.no_push:
            continue

        if ckpt.is_done("pushed", lc):
            print(f"[OK] {lc}: already pushed")
            continue

        ok, repo_used = push_with_fallback(
            ds=ds,
            lang_code=lc,
            repo_id=args.repo_id,
            hf_token=hf_token,
            backup_repo_id=args.backup_repo_id,
            max_shard_size=args.max_shard_size,
        )
        if ok:
            ckpt.set_done("pushed", lc, {"repo": repo_used, "rows": len(ds), "ts": time.time()})
        else:
            print(f"[FAIL] {lc}: push failed (kept local at {save_dir})")

    print("\nDONE.")


if __name__ == "__main__":
    main()