"""Shared training loop for every Project 2.2 model.

Usage (from the repo root, e.g. in Colab):
    python -m src.train --config configs/default.yaml --model all --skip-existing
    python -m src.train --config configs/default.yaml --model gated_clip --seeds 0
    add --cache-dir /content/features_local in Colab to copy features off Drive first (faster)

Each run writes to <runs_dir>/<model>/seed<k>/:
    model.pt          best weights (selected on dev AUPRC)
    history.csv       per-epoch train loss, dev loss and dev AUPRC (for overfitting curves)
    predictions.csv   pair_id, split, model, seed, y, p_unanswerable (+ gate_q for gated fusion), dev and test
    summary.json      settings, parameter count, epochs, time per epoch, peak GPU memory
The test split is only scored, never used for selection or early stopping.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score

from src.data import load_features
from src.models import build_model

TRAINABLE = ["question_only", "image_only", "early", "clip_early", "cross_attention", "gated", "gated_clip"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_dir(cfg, model_name, seed):
    d = Path(cfg["paths"]["runs_dir"]) / model_name / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_loss(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


@torch.no_grad()
def predict(model, store, idx, device, bs=1024):
    """Probability of 'unanswerable' for each row in idx (and the gate value, for gated fusion)."""
    model.eval()
    probs, gates = [], []
    for i in range(0, len(idx), bs):
        logits = model(store.batch(idx[i:i + bs], model.needs, device))
        probs.append(torch.sigmoid(logits).float().cpu())
        if getattr(model, "last_gate_q", None) is not None:
            gates.append(model.last_gate_q.float().cpu())
    probs = torch.cat(probs).numpy()
    return probs, (torch.cat(gates).numpy() if gates else None)


def train_one(store, cfg, model_name, seed, device):
    """Train one model with one seed; early-stop on dev AUPRC; save everything needed downstream."""
    set_seed(seed)
    tcfg = cfg["training"]
    if model_name not in cfg["models"]:
        raise ValueError(f"'{model_name}' has no entry under models: in the config")
    hp = cfg["models"][model_name]
    model = build_model(model_name, store.dims, **hp).to(device)
    tr, dev, test = store.index("train"), store.index("dev"), store.index("test")
    target = torch.from_numpy(store.meta["target"].values.astype(np.float32))
    y = store.meta["y"].values

    pos = float(y[tr].mean())  # class imbalance: weight the minority (unanswerable) class
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    rng = np.random.default_rng(seed)
    probe = rng.choice(tr, size=min(len(tr), tcfg["train_probe_size"]), replace=False)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best_ap, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // tcfg["batch_size"])):
            loss = loss_fn(model(store.batch(b, model.needs, device)), target[torch.as_tensor(b)].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        secs = time.time() - t0
        p_probe, _ = predict(model, store, probe, device)
        p_dev, _ = predict(model, store, dev, device)
        ap = float(average_precision_score(y[dev], p_dev))  # selection metric (Saito & Rehmsmeier 2015)
        history.append({"epoch": epoch, "train_loss": log_loss(p_probe, y[probe]),
                        "dev_loss": log_loss(p_dev, y[dev]), "dev_auprc": ap, "seconds": round(secs, 2)})
        print(f"[{model_name} seed {seed}] epoch {epoch:2d}  dev AUPRC {ap:.4f}  ({secs:.1f}s)")
        if ap > best_ap:
            best_ap, best_state, best_epoch, bad = ap, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    model.load_state_dict(best_state)

    out = run_dir(cfg, model_name, seed)
    torch.save(model.state_dict(), out / "model.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    frames = []
    for split, idx in (("dev", dev), ("test", test)):
        if len(idx) == 0:
            continue
        p, gate = predict(model, store, idx, device)
        y_split = y[idx] if np.isnan(y[idx]).any() else y[idx].astype(int)  # unlabelled rows stay empty
        f = pd.DataFrame({"pair_id": store.meta["pair_id"].values[idx], "split": split, "model": model_name,
                          "seed": seed, "y": y_split, "p_unanswerable": p})
        if gate is not None:
            f["gate_q"] = gate
        frames.append(f)
    pd.concat(frames).to_csv(out / "predictions.csv", index=False)
    summary = {"model": model_name, "seed": seed, "label": cfg["data"]["label"], "hyperparameters": hp,
               "params": int(sum(p.numel() for p in model.parameters())), "epochs_run": len(history),
               "best_epoch": best_epoch, "best_dev_auprc": round(best_ap, 4),
               "sec_per_epoch": round(float(np.mean([h["seconds"] for h in history])), 2),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return out


def finished(cfg, model_name, seed):
    """True if this model and seed already have dev and test predictions."""
    f = Path(cfg["paths"]["runs_dir"]) / model_name / f"seed{seed}" / "predictions.csv"
    return f.exists() and {"dev", "test"} <= set(pd.read_csv(f, usecols=["split"])["split"])


def late_fusion(cfg, seed):
    """Weighted average of the question-only and image-only probabilities; weight chosen on dev AUPRC."""
    q = pd.read_csv(run_dir(cfg, "question_only", seed) / "predictions.csv")
    v = pd.read_csv(run_dir(cfg, "image_only", seed) / "predictions.csv")
    m = q.merge(v[["pair_id", "split", "p_unanswerable"]], on=["pair_id", "split"], suffixes=("_q", "_v"))
    dev = m[m.split == "dev"]
    weights = np.linspace(0, 1, 21)
    scores = [average_precision_score(dev.y, w * dev.p_unanswerable_q + (1 - w) * dev.p_unanswerable_v) for w in weights]
    w = float(weights[int(np.argmax(scores))])
    out = run_dir(cfg, "late", seed)
    pd.DataFrame({"pair_id": m.pair_id, "split": m.split, "model": "late", "seed": seed, "y": m.y,
                  "p_unanswerable": w * m.p_unanswerable_q + (1 - w) * m.p_unanswerable_v}
                 ).to_csv(out / "predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps(
        {"model": "late", "seed": seed, "question_weight": w, "best_dev_auprc": round(float(max(scores)), 4)}, indent=2))
    print(f"[late seed {seed}] question weight {w:.2f}  dev AUPRC {max(scores):.4f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train Project 2.2 models")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="all", help="one of %s, late, or all" % ", ".join(TRAINABLE))
    ap.add_argument("--seeds", type=int, nargs="*", help="override the seeds in the config")
    ap.add_argument("--parts", nargs="*", help="override data.parts (only for the older per-part layout)")
    ap.add_argument("--skip-existing", action="store_true", help="skip model/seed runs that already have predictions")
    ap.add_argument("--cache-dir", help="copy feature files here before loading (e.g. /content/features_local)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.parts:
        cfg["data"]["parts"] = args.parts
    seeds = args.seeds if args.seeds else cfg["training"]["seeds"]
    names = TRAINABLE + ["late"] if args.model == "all" else [args.model]
    unknown = [n for n in names if n != "late" and n not in cfg["models"]]
    if unknown:
        raise SystemExit(f"no config entry under models: for {unknown}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = cfg["data"]
    cache_dir = args.cache_dir or cfg["paths"].get("local_cache_dir")
    store = None
    for seed in seeds:
        for name in names:
            if args.skip_existing and finished(cfg, name, seed):
                print(f"[{name} seed {seed}] already has predictions, skipped")
                continue
            if name == "late":
                late_fusion(cfg, seed)
                continue
            if store is None:
                store = load_features(cfg["paths"]["features_dir"], data.get("parts"), data["label"],
                                      data.get("soft_label_weight", 1.0), cache_dir)
                print({s: int(len(store.index(s))) for s in ("train", "dev", "test")}, "pairs | device:", device)
            train_one(store, cfg, name, seed, device)


if __name__ == "__main__":
    main()
