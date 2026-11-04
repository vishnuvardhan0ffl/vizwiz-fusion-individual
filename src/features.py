"""Download VizWiz-VQA and extract frozen features for every split.

Run from the repo root in Colab (T4 GPU):
    python -m src.features --data-dir /content/data --out /content/features
Writes <out>/meta.csv, <out>/manifest.json and one .npy per array (format: docs/data_contract.md).

Encoders (frozen): ViT-B/16 pretrained on ImageNet-21k (Dosovitskiy et al. 2021), BERT-base uncased
(Devlin et al. 2019) and CLIP ViT-B/32 (Radford et al. 2021), loaded with Hugging Face Transformers
(Wolf et al. 2020). Each photo is decoded once (EXIF rotation, 224x224 resize without cropping) and fed to
both image encoders. Photo sharpness is the variance of the Laplacian (Pech-Pacheco et al. 2000).
"""
import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

BASE = "https://vizwiz.cs.colorado.edu"
DOWNLOADS = {
    "Annotations.zip": f"{BASE}/VizWiz_final/vqa_data/Annotations.zip",
    "train.zip": f"{BASE}/VizWiz_final/images/train.zip",
    "val.zip": f"{BASE}/VizWiz_final/images/val.zip",
    "test.zip": f"{BASE}/VizWiz_final/images/test.zip",
    "VQA_test.json": f"{BASE}/VizWiz_all_answers/VQA_test.json",   # labelled test split (released April 2026)
}
IMAGE_MODEL = "google/vit-base-patch16-224-in21k"
TEXT_MODEL = "bert-base-uncased"
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
SIZE, GRID, MAX_Q = 224, 7, 40
UNANS = {"unanswerable", "unsuitable", "unsuitable image"}


# ---------------------------------------------------------------- download
def download(url, path, chunk=8 * 1024 * 1024):
    """Stream a file to disk, resuming a partial download when the server allows it."""
    done = path.stat().st_size if path.exists() else 0
    with requests.get(url, stream=True, headers={"Range": f"bytes={done}-"} if done else {}, timeout=60) as r:
        if done and r.status_code != 206:
            done = 0
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + done
        with open(path, "ab" if done else "wb") as f, tqdm(total=total, initial=done, unit="B",
                                                          unit_scale=True, desc=path.name) as bar:
            for block in r.iter_content(chunk):
                f.write(block)
                bar.update(len(block))


def fetch_all(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        marker = data_dir / f".{name}.done"
        if marker.exists():
            continue
        target = data_dir / name
        download(url, target)
        if name.endswith(".zip"):
            subprocess.run(["unzip", "-q", "-o", str(target), "-d", str(data_dir)], check=True)
            target.unlink()
        marker.touch()


# ---------------------------------------------------------------- labels and meta
def find(root, filename):
    hits = sorted(p for p in root.rglob(filename) if "__MACOSX" not in p.parts)
    if not hits:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return hits[0]


def read_split(path, split):
    rows = []
    for r in json.loads(Path(path).read_text()):
        answers = [str(a.get("answer", "")).strip().lower() for a in r.get("answers", [])]
        votes = sum(a in UNANS for a in answers)
        if "answerable" in r:
            y = 1 - int(r["answerable"])
        elif answers:                       # fallback: majority of crowd answers say unanswerable
            y = int(votes > len(answers) / 2)
        else:
            y = np.nan
        rows.append({"pair_id": r["image"], "question": str(r["question"]).strip(), "official_split": split,
                     "y": y, "unans_votes": votes if answers else np.nan,
                     "answer_type": r.get("answer_type", "unknown")})
    return pd.DataFrame(rows)


def build_meta(data_dir):
    meta = pd.concat([read_split(find(data_dir, "train.json"), "train"),
                      read_split(find(data_dir, "val.json"), "val"),
                      read_split(data_dir / "VQA_test.json", "test")], ignore_index=True)
    paths = {}
    for p in data_dir.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "__MACOSX" not in p.parts:
            paths[p.name] = str(p)
    meta["path"] = meta["pair_id"].map(paths)
    missing = int(meta["path"].isna().sum())
    meta = meta.dropna(subset=["path"]).reset_index(drop=True)
    # how often each question text occurs in training: rare questions test reliance on language priors
    freq = Counter(meta.loc[meta.official_split == "train", "question"].str.lower())
    meta["q_freq_train"] = meta["question"].str.lower().map(freq).fillna(0).astype(int)
    meta["rare_question"] = meta["q_freq_train"] <= 1
    meta["q_words"] = meta["question"].str.split().str.len()
    return meta, missing


# ---------------------------------------------------------------- photos
class Photos(Dataset):
    """Decoded photo in [0, 1] (3 x 224 x 224) plus a readable flag, sharpness and brightness."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.w = np.array([0.299, 0.587, 0.114], np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                im.draft("RGB", (SIZE, SIZE))
                im = ImageOps.exif_transpose(im).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC)
            a = np.asarray(im, dtype=np.float32) / 255.0
            g = (a @ self.w) * 255.0
            lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
            return torch.from_numpy(a).permute(2, 0, 1), 1, float(lap.var()), float(g.mean())
        except Exception:
            return torch.zeros(3, SIZE, SIZE), 0, float("nan"), float("nan")


def norm(x, mean, std):
    return (x - torch.tensor(mean, device=x.device).view(1, 3, 1, 1)) / torch.tensor(std, device=x.device).view(1, 3, 1, 1)


def peak_gb(device):
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None


@torch.no_grad()
def extract_images(paths, device, bs, workers):
    from transformers import AutoImageProcessor, AutoModel, CLIPModel
    proc = AutoImageProcessor.from_pretrained(IMAGE_MODEL)
    vit = AutoModel.from_pretrained(IMAGE_MODEL).to(device).eval()
    clip = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    n, d = len(paths), vit.config.hidden_size
    d_clip = clip.config.projection_dim
    out = {"img_cls": np.empty((n, d), np.float16), "img_tok": np.empty((n, GRID * GRID, d), np.float16),
           "clip_img": np.empty((n, d_clip), np.float16), "img_ok": np.empty(n, np.uint8)}
    sharp, bright = np.empty(n, np.float32), np.empty(n, np.float32)
    amp = dict(device_type=device, dtype=torch.float16, enabled=(device == "cuda"))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, i = time.time(), 0
    for x, ok, s, b in tqdm(DataLoader(Photos(paths), batch_size=bs, num_workers=workers), desc="photos"):
        x, k = x.to(device, non_blocking=True), len(x)
        with torch.autocast(**amp):
            h = vit(pixel_values=norm(x, proc.image_mean, proc.image_std)).last_hidden_state
            c = clip.visual_projection(clip.vision_model(pixel_values=norm(x, CLIP_MEAN, CLIP_STD)).pooler_output)
        side = int(round((h.shape[1] - 1) ** 0.5))
        grid = h[:, 1:].float().reshape(k, side, side, d).permute(0, 3, 1, 2)
        out["img_cls"][i:i + k] = h[:, 0].float().cpu().numpy()
        out["img_tok"][i:i + k] = F.adaptive_avg_pool2d(grid, GRID).flatten(2).transpose(1, 2).cpu().numpy()
        out["clip_img"][i:i + k] = F.normalize(c.float(), dim=-1).cpu().numpy()
        out["img_ok"][i:i + k], sharp[i:i + k], bright[i:i + k] = ok.numpy(), s.numpy(), b.numpy()
        i += k
    secs = time.time() - t0
    timing = {"n": n, "minutes": round(secs / 60, 1), "images_per_sec": round(n / secs, 1),
              "peak_vram_gb": peak_gb(device), "unreadable": int((out["img_ok"] == 0).sum())}
    return out, sharp, bright, timing


@torch.no_grad()
def extract_text(questions, device, bs):
    from transformers import AutoModel, AutoTokenizer, CLIPModel
    tok, bert = AutoTokenizer.from_pretrained(TEXT_MODEL), AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
    ctok, clip = AutoTokenizer.from_pretrained(CLIP_MODEL), CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    n, d, d_clip = len(questions), bert.config.hidden_size, clip.config.projection_dim
    out = {"txt_cls": np.empty((n, d), np.float16), "txt_mean": np.empty((n, d), np.float16),
           "txt_tok": np.empty((n, MAX_Q, d), np.float16), "txt_mask": np.empty((n, MAX_Q), np.uint8),
           "clip_txt": np.empty((n, d_clip), np.float16)}
    amp = dict(device_type=device, dtype=torch.float16, enabled=(device == "cuda"))
    truncated = 0
    t0 = time.time()
    for i in tqdm(range(0, n, bs), desc="questions"):
        qs = questions[i:i + bs]
        truncated += sum(len(t) > MAX_Q for t in tok(qs)["input_ids"])
        enc = tok(qs, padding="max_length", truncation=True, max_length=MAX_Q, return_tensors="pt").to(device)
        cenc = ctok(qs, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.autocast(**amp):
            h = bert(**enc).last_hidden_state.float()
            c = clip.text_projection(clip.text_model(**cenc).pooler_output)
        m = enc["attention_mask"].unsqueeze(-1).float()
        k = len(qs)
        out["txt_cls"][i:i + k] = h[:, 0].cpu().numpy()
        out["txt_mean"][i:i + k] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
        out["txt_tok"][i:i + k] = h.cpu().numpy()
        out["txt_mask"][i:i + k] = enc["attention_mask"].cpu().numpy()
        out["clip_txt"][i:i + k] = F.normalize(c.float(), dim=-1).cpu().numpy()
    return out, {"n": n, "minutes": round((time.time() - t0) / 60, 1), "truncated_share": round(truncated / n, 4)}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Download VizWiz-VQA and extract features")
    ap.add_argument("--data-dir", default="/content/data")
    ap.add_argument("--out", default="/content/features")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args(argv)
    data_dir, out = Path(args.data_dir), Path(args.out)
    if (out / "manifest.json").exists():
        print(f"{out}/manifest.json exists: features already extracted, nothing to do")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if not args.skip_download:
        fetch_all(data_dir)
    meta, missing = build_meta(data_dir)
    print(f"{len(meta):,} pairs with a photo ({missing} without); splits {meta.official_split.value_counts().to_dict()}")

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"n_rows": len(meta), "meta": "meta.csv", "features": {},
                "encoders": {"image": IMAGE_MODEL, "text": TEXT_MODEL, "clip": CLIP_MODEL}, "timing": {}}

    def save(arrays):
        for name, arr in arrays.items():
            np.save(out / f"{name}.npy", arr)
            manifest["features"][name] = {"file": f"{name}.npy", "shape": list(arr.shape), "dtype": str(arr.dtype)}

    imgs, sharp, bright, manifest["timing"]["images"] = extract_images(meta["path"].tolist(), device,
                                                                       args.batch_size, args.workers)
    save(imgs)
    del imgs
    txt, manifest["timing"]["questions"] = extract_text(meta["question"].tolist(), device, 2 * args.batch_size)
    save(txt)
    del txt

    meta["sharpness"], meta["brightness"] = sharp, bright
    cuts = np.nanquantile(meta.loc[meta.official_split == "train", "sharpness"], [1 / 3, 2 / 3])
    meta["sharp_bin"] = pd.cut(meta["sharpness"], [-np.inf, cuts[0], cuts[1], np.inf],
                               labels=["blurry", "medium", "sharp"]).astype(str)
    meta.to_csv(out / "meta.csv", index=False)   # path kept for the failure-case gallery
    manifest["timing"]["total_minutes"] = round((time.time() - t0) / 60, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["timing"], indent=2))


if __name__ == "__main__":
    main()
