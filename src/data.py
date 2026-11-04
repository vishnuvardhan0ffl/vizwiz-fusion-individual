"""Load cached features and assign the Project 2.2 splits.

Two layouts are supported (details in docs/data_contract.md):
  1. Hand-off layout (default): features_dir/manifest.json + meta.csv + one .npy per array, written by
     notebooks/02_data_and_baselines.ipynb for all splits at once.
  2. Part layout (older): features_dir/<part>/meta.csv + <name>.npy, listed in data.parts.
Splits follow the official VizWiz partition: official train -> train, validation -> dev (tuning),
test -> test. Arrays are loaded only when a model first needs them, so models that use pooled
vectors never read the large token arrays.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REQUIRED_META = ("pair_id", "question", "official_split", "y", "unans_votes")
SPLIT_MAP = {"train": "train", "val": "dev", "test": "test"}
N_CROWD_ANSWERS = 10

# batch key -> feature array name
ARRAY_FOR_KEY = {
    "q_vec": "txt_mean",    # BERT mask-aware mean
    "v_vec": "img_cls",     # ViT [CLS]
    "q_tok": "txt_tok",     # BERT token states
    "q_mask": "txt_mask",   # BERT attention mask
    "v_tok": "img_tok",     # ViT 7x7 region tokens
    "q_clip": "clip_txt",   # CLIP text embedding
    "v_clip": "clip_img",   # CLIP image embedding
}


class FeatureStore:
    """Meta (one row per pair) plus feature arrays that are read from disk on first use."""

    def __init__(self, meta, files, cache_dir=None):
        self.meta = meta
        self.files = files            # array name -> list of (part, path); parts are stacked in order
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._arrays = {}

    def index(self, split):
        """Row positions of one split ('train', 'dev' or 'test')."""
        return np.where(self.meta["split"].values == split)[0]

    def has(self, name):
        return name in self.files

    def width(self, name):
        """Feature size of a 2-D array, read from the file header only."""
        _, path = self.files[name][0]
        return int(np.load(path, mmap_mode="r").shape[-1])

    @property
    def dims(self):
        """Input sizes for model construction: BERT/ViT always, CLIP when the hand-off has it."""
        d = {"d_q": self.width("txt_mean"), "d_v": self.width("img_cls")}
        if self.has("clip_txt") and self.has("clip_img"):
            d.update(d_q_clip=self.width("clip_txt"), d_v_clip=self.width("clip_img"))
        return d

    def _local(self, part, path):
        """Copy a file to the local cache (much faster than Drive in Colab); skip if already there."""
        if self.cache_dir is None:
            return path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dst = self.cache_dir / (f"{part}__{path.name}" if part else path.name)
        if not dst.exists() or dst.stat().st_size != path.stat().st_size:
            shutil.copyfile(path, dst)
        return dst

    def array(self, name):
        if name not in self.files:
            raise KeyError(f"feature '{name}' is not in the hand-off (available: {sorted(self.files)})")
        if name not in self._arrays:
            parts = [np.load(self._local(part, path)) for part, path in self.files[name]]
            arr = parts[0] if len(parts) == 1 else np.concatenate(parts)
            self._arrays[name] = torch.from_numpy(np.ascontiguousarray(arr))
        return self._arrays[name]

    def batch(self, idx, needs, device):
        """Build the inputs a model asks for (its `needs`), moved to `device`."""
        idx = torch.as_tensor(np.asarray(idx))
        out = {}
        for key in needs:
            a = self.array(ARRAY_FOR_KEY[key])[idx].to(device)
            out[key] = a.bool() if key == "q_mask" else a.float()
        return out


def _rows(path):
    return int(np.load(path, mmap_mode="r").shape[0])


def _read_manifest(features_dir):
    manifest = json.loads((features_dir / "manifest.json").read_text())
    meta = pd.read_csv(features_dir / manifest.get("meta", "meta.csv"))
    files = {name: [("", features_dir / info["file"])] for name, info in manifest["features"].items()}
    return [meta], files


def _read_parts(features_dir, parts):
    if not parts:
        raise ValueError(f"no manifest.json in {features_dir} and data.parts is empty")
    metas, files = [], {}
    for part in parts:
        folder = features_dir / part
        meta = pd.read_csv(folder / "meta.csv").assign(part=part)
        metas.append(meta)
        for path in sorted(folder.glob("*.npy")):
            files.setdefault(path.stem, []).append((part, path))
    # an array is usable only if every part has it
    files = {k: v for k, v in files.items() if len(v) == len(parts)}
    return metas, files


def load_features(features_dir, parts=None, label="hard", soft_weight=1.0, cache_dir=None):
    """Read meta and register the feature files; check that rows and splits are consistent."""
    features_dir = Path(features_dir)
    if (features_dir / "manifest.json").exists():
        metas, files = _read_manifest(features_dir)
    else:
        metas, files = _read_parts(features_dir, parts)

    for i, meta in enumerate(metas):
        missing = [c for c in REQUIRED_META if c not in meta.columns]
        if missing:
            raise ValueError(f"meta.csv is missing columns: {missing}")
        for name, entries in files.items():
            n = _rows(entries[i][1])
            if n != len(meta):
                raise ValueError(f"{entries[i][1]} has {n} rows but its meta.csv has {len(meta)}")

    meta = pd.concat(metas, ignore_index=True)
    meta["split"] = meta["official_split"].map(SPLIT_MAP)
    if meta["split"].isna().any():
        bad = sorted(meta.loc[meta["split"].isna(), "official_split"].astype(str).unique())
        raise ValueError(f"unknown official_split values: {bad}")
    if not meta["pair_id"].is_unique:
        raise ValueError("pair_id must be unique")

    meta["y"] = meta["y"].astype(np.float32)
    if label == "hard":
        meta["target"] = meta["y"]
    elif label == "soft":
        # soft label: (1 - w) * hard label + w * share of the 10 crowd answers saying "unanswerable"
        share = (meta["unans_votes"] / N_CROWD_ANSWERS).clip(0, 1)
        meta["target"] = ((1 - soft_weight) * meta["y"] + soft_weight * share).astype(np.float32)
    else:
        raise ValueError("data.label must be 'hard' or 'soft'")
    return FeatureStore(meta=meta, files=files, cache_dir=cache_dir)
