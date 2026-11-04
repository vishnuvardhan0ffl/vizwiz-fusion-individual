"""End-to-end check of the whole pipeline on fake data with stub encoders (CPU, no downloads, a few minutes).
Run: python -m tests.test_pipeline
It covers src.features (meta, photo quality, ViT/BERT/CLIP arrays, manifest), src.train and src.evaluate.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageFilter

D, D_CLIP = 32, 16


def fake_data(root, rng):
    """Photos and VizWiz-style annotations; unanswerable photos are blurred so sharpness carries signal."""
    (root / "Annotations").mkdir(parents=True)
    for split, n, fname in (("train", 240, "Annotations/train.json"), ("val", 80, "Annotations/val.json"),
                            ("test", 80, "VQA_test.json")):
        (root / split).mkdir()
        recs = []
        for i in range(n):
            unans = rng.random() < 0.3
            name = f"VizWiz_{split}_{i:08d}.jpg"
            img = Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
            (img.filter(ImageFilter.GaussianBlur(5)) if unans else img).save(root / split / name)
            q = "what is this" if unans else rng.choice(["what color is this", "read this label please"])
            ans = [{"answer": "unanswerable" if (unans and rng.random() < 0.8) else "blue"} for _ in range(10)]
            recs.append({"image": name, "question": q, "answers": ans, "answer_type": "other",
                         **({"answerable": 0 if unans else 1} if split != "test" or i % 2 else {})})
        (root / fname).write_text(json.dumps(recs))     # half the test records lack "answerable": tests the fallback


class Tok:
    def __call__(self, texts, padding=False, truncation=False, max_length=None, return_tensors=None):
        from transformers import BatchEncoding
        ids = [[101] + [sum(map(ord, w)) % 3000 + 5 for w in t.split()] + [102] for t in texts]
        if truncation and max_length:
            ids = [x[:max_length] for x in ids]
        L = max_length if padding == "max_length" else max(map(len, ids))
        mask = [[1] * len(x) + [0] * (L - len(x)) for x in ids] if padding else [[1] * len(x) for x in ids]
        ids = [x + [0] * (L - len(x)) for x in ids] if padding else ids
        d = {"input_ids": ids, "attention_mask": mask}
        return BatchEncoding({k: torch.tensor(v) for k, v in d.items()} if return_tensors == "pt" else d)


class Vit(nn.Module):
    def __init__(self):
        super().__init__()
        self.config, self.p = SimpleNamespace(hidden_size=D), nn.Conv2d(3, D, 16, 16)

    def forward(self, pixel_values):
        x = self.p(pixel_values).flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=torch.cat([x.mean(1, keepdim=True), x], 1))


class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.config, self.e = SimpleNamespace(hidden_size=D), nn.Embedding(4000, D)

    def forward(self, input_ids, attention_mask=None, **kw):
        return SimpleNamespace(last_hidden_state=self.e(input_ids % 4000))


class Clip(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(projection_dim=D_CLIP)
        self.v, self.t = nn.Conv2d(3, D, 32, 32), nn.Embedding(4000, D)
        self.visual_projection, self.text_projection = nn.Linear(D, D_CLIP), nn.Linear(D, D_CLIP)

    def vision_model(self, pixel_values):
        return SimpleNamespace(pooler_output=self.v(pixel_values).mean((2, 3)))

    def text_model(self, input_ids, attention_mask=None, **kw):
        return SimpleNamespace(pooler_output=self.t(input_ids % 4000).mean(1))


def patch_transformers():
    import transformers
    transformers.AutoTokenizer.from_pretrained = staticmethod(lambda name, **k: Tok())
    transformers.AutoImageProcessor.from_pretrained = staticmethod(
        lambda name, **k: SimpleNamespace(image_mean=[0.5] * 3, image_std=[0.5] * 3))
    transformers.AutoModel.from_pretrained = staticmethod(lambda name, **k: Vit() if "vit" in name else Bert())
    transformers.CLIPModel.from_pretrained = staticmethod(lambda name, **k: Clip())


def run():
    patch_transformers()
    from src import evaluate, features, train
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake_data(tmp / "data", rng)
        features.main(["--data-dir", str(tmp / "data"), "--out", str(tmp / "features"), "--skip-download",
                       "--batch-size", "32", "--workers", "0"])
        man = json.loads((tmp / "features" / "manifest.json").read_text())
        assert {"img_cls", "img_tok", "clip_img", "txt_mean", "txt_tok", "txt_mask", "clip_txt"} <= set(man["features"])
        import pandas as pd
        meta = pd.read_csv(tmp / "features" / "meta.csv")
        assert len(meta) == 400 and meta.y.notna().all() and set(meta.sharp_bin) == {"blurry", "medium", "sharp"}
        assert meta.groupby("y").sharpness.median()[1] < meta.groupby("y").sharpness.median()[0]   # blur detected

        for name, runs in (("default", "runs"), ("ablation", "runs_ablation")):
            cfg = yaml.safe_load(open(f"configs/{name}.yaml"))
            cfg["paths"] = {"features_dir": str(tmp / "features"), "runs_dir": str(tmp / runs), "local_cache_dir": None}
            cfg["training"].update(epochs=2, batch_size=64, train_probe_size=100, seeds=[0, 1])
            (tmp / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
        train.main(["--config", str(tmp / "default.yaml"), "--model", "all"])
        for m in ["gated_md0", "cross_attention_l2"]:
            train.main(["--config", str(tmp / "ablation.yaml"), "--model", m])

        evaluate.main(["--config", str(tmp / "default.yaml"), "--ablation-config", str(tmp / "ablation.yaml"),
                       "--out", str(tmp / "results")])
        res = tmp / "results"
        summary = json.loads((res / "summary.json").read_text())
        for f in ["test_metrics_summary.csv", "test_bootstrap_ci_seed0.csv", "fusion_gain_vs_best_unimodal.csv",
                  "mcnemar_holm.csv", "subgroups.csv", "gate_by_sharpness.csv", "modality_ablation.csv",
                  "efficiency.csv", "failure_cases.csv", "data_quality.csv", "ablation.csv"]:
            assert (res / "tables" / f).exists(), f
        for f in ["data_quality.png", "model_comparison.png", "roc_pr_calibration.png", "subgroups_sharpness.png",
                  "learning_curves.png", "modality_ablation.png", "failure_cases.png"]:
            assert (res / "figures" / f).exists(), f
        assert "gated_clip" in summary["test_summary_mean"] and "majority" in summary["test_summary_mean"]
        assert summary["efficiency"] and summary["gate"]
    print("PIPELINE TEST PASSED")


if __name__ == "__main__":
    run()
