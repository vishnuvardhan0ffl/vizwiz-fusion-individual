"""Load trained models from their run folders, and score them with one modality removed.

Every run of `python -m src.train` saves runs/<model>/seed<k>/model.pt plus summary.json, which records the
exact hyperparameters used. `load_run` rebuilds the model from those and loads the weights, so evaluation,
latency timing and ablations can use the trained models without retraining.

Modality ablation: `predict_without` replaces the question or the image inputs with their training-set mean,
so the model receives no information from that modality while the input shape stays the same.

Example (Colab, repo root, Drive mounted):
    from src.train import load_config
    from src.data import load_features
    from src.load import load_run, input_means, predict_without
    cfg = load_config("configs/default.yaml")
    store = load_features(cfg["paths"]["features_dir"])
    model = load_run(cfg, "gated", seed=0, store=store, device="cuda")
    test = store.index("test")
    means = input_means(store, model.needs, store.index("train"))
    p_no_image = predict_without(model, store, test, "image", means, device="cuda")
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.data import ARRAY_FOR_KEY
from src.models import build_model

QUESTION_KEYS = {"q_vec", "q_tok", "q_clip"}
IMAGE_KEYS = {"v_vec", "v_tok", "v_clip"}


def load_run(cfg, model_name, seed, store, device="cpu", runs_dir=None):
    """Rebuild model_name with the hyperparameters saved in its summary.json and load its best weights."""
    run = Path(runs_dir or cfg["paths"]["runs_dir"]) / model_name / f"seed{seed}"
    weights = run / "model.pt"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found (late fusion and the floor models have no weights)")
    summary = run / "summary.json"
    if summary.exists():
        hp = json.loads(summary.read_text()).get("hyperparameters", {})
    else:
        hp = cfg["models"][model_name]
    model = build_model(model_name, store.dims, **hp)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def input_means(store, needs, idx, bs=2048):
    """Training-set mean of each input the model uses. Region tokens keep their position (shape 49 x D);
    question tokens are averaged over real tokens only (shape D), so padding does not drag the mean."""
    means = {}
    for key in needs:
        if key == "q_mask":
            continue
        arr = store.array(ARRAY_FOR_KEY[key])
        total, count = 0.0, 0.0
        for i in range(0, len(idx), bs):
            b = torch.as_tensor(np.asarray(idx[i:i + bs]))
            x = arr[b].float()
            if key == "q_tok":
                m = store.array("txt_mask")[b].float().unsqueeze(-1)
                total, count = total + (x * m).sum((0, 1)), count + m.sum()
            else:
                total, count = total + x.sum(0), count + len(b)
        means[key] = total / count
    return means


@torch.no_grad()
def predict_without(model, store, idx, modality, means, device="cpu", bs=1024):
    """Probability of 'unanswerable' with the question or the image replaced by its training mean.
    modality: "question", "image" or None (no replacement). The question mask is kept for cross-attention,
    so question length still reaches that model; mention this when reporting."""
    keys = {"question": QUESTION_KEYS, "image": IMAGE_KEYS, None: set()}[modality]
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        batch = store.batch(idx[i:i + bs], model.needs, device)
        for key in keys & set(batch):
            batch[key] = means[key].to(device).expand_as(batch[key]).clone()
        out.append(torch.sigmoid(model(batch)).float().cpu())
    return torch.cat(out).numpy()
