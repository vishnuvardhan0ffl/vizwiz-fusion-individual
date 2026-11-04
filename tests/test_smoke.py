"""Fast end-to-end check on synthetic features (CPU, about a minute). Run: python -m tests.test_smoke"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.config_table import collect, table
from src.data import load_features
from src.load import input_means, load_run, predict_without
from src.train import TRAINABLE, load_config, main

D, D_CLIP, N_TOK, N_REG = 32, 16, 40, 49


def synthetic(splits, prefix, rng, clip=True):
    """Meta and arrays whose label leaks weakly into every input, so all models can learn something."""
    n = len(splits)
    y = (rng.random(n) < 0.3).astype(int)
    meta = pd.DataFrame({"pair_id": [f"{prefix}_{i:05d}.jpg" for i in range(n)], "question": "what is this?",
                         "official_split": splits, "y": y,
                         "unans_votes": np.clip(y * 7 + rng.integers(0, 3, n), 0, 10)})
    sig = lambda d: (y[:, None] - 0.3) * 0.8 * np.random.default_rng(d).choice([-1.0, 1.0], d)  # survives LayerNorm
    mask = np.zeros((n, N_TOK), np.uint8)
    mask[:, :8] = 1
    arrays = {"img_cls": rng.normal(size=(n, D)) + sig(D), "img_tok": rng.normal(size=(n, N_REG, D)) + sig(D)[:, None],
              "txt_mean": rng.normal(size=(n, D)) + sig(D), "txt_tok": rng.normal(size=(n, N_TOK, D)) + sig(D)[:, None],
              "txt_mask": mask}
    if clip:
        arrays.update(clip_img=rng.normal(size=(n, D_CLIP)) + sig(D_CLIP), clip_txt=rng.normal(size=(n, D_CLIP)) + sig(D_CLIP))
    return meta, {k: a.astype(np.uint8 if k == "txt_mask" else np.float16) for k, a in arrays.items()}


def write_handoff(folder, rng):
    """Hand-off layout, as written by notebooks/02_data_and_baselines.ipynb."""
    folder.mkdir(parents=True)
    meta, arrays = synthetic(["train"] * 300 + ["val"] * 80 + ["test"] * 100, "pair", rng)
    meta.to_csv(folder / "meta.csv", index=False)
    manifest = {"n_rows": len(meta), "meta": "meta.csv", "features": {}}
    for k, a in arrays.items():
        np.save(folder / f"{k}_fp123.npy", a)
        manifest["features"][k] = {"file": f"{k}_fp123.npy", "shape": list(a.shape), "dtype": str(a.dtype)}
    (folder / "manifest.json").write_text(json.dumps(manifest))


def write_parts(folder, rng):
    """Older per-part layout (no CLIP)."""
    for part, splits in (("trainval", ["train"] * 300 + ["val"] * 80), ("test", ["test"] * 100)):
        (folder / part).mkdir(parents=True)
        meta, arrays = synthetic(splits, part, rng, clip=False)
        meta.to_csv(folder / part / "meta.csv", index=False)
        for k, a in arrays.items():
            np.save(folder / part / f"{k}.npy", a)


def config(tmp, features_dir, runs_dir, **data):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["paths"] = {"features_dir": str(features_dir), "runs_dir": str(runs_dir), "local_cache_dir": None}
    cfg["data"].update(data)
    cfg["training"].update(epochs=2, batch_size=64, train_probe_size=100)
    path = tmp / f"cfg_{Path(runs_dir).name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def run():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. hand-off layout: every model, then a rerun that must skip everything
        write_handoff(tmp / "handoff", rng)
        cfg = config(tmp, tmp / "handoff", tmp / "runs")
        main(["--config", cfg, "--model", "all", "--seeds", "0"])
        for name in TRAINABLE + ["late"]:
            pred = pd.read_csv(tmp / "runs" / name / "seed0" / "predictions.csv")
            assert set(pred.split) == {"dev", "test"} and len(pred) == 180, name
            assert {"pair_id", "split", "model", "seed", "y", "p_unanswerable"} <= set(pred.columns), name
            assert pred.p_unanswerable.between(0, 1).all(), name
        for name in ("gated", "gated_clip"):
            assert "gate_q" in pd.read_csv(tmp / "runs" / name / "seed0" / "predictions.csv").columns, name
        summary = json.loads((tmp / "runs" / "clip_early" / "seed0" / "summary.json").read_text())
        assert summary["hyperparameters"]["features"] == "clip" and summary["params"] > 0
        before = (tmp / "runs" / "gated" / "seed0" / "predictions.csv").stat().st_mtime_ns
        main(["--config", cfg, "--model", "all", "--seeds", "0", "--skip-existing"])
        assert (tmp / "runs" / "gated" / "seed0" / "predictions.csv").stat().st_mtime_ns == before

        # 1b. reload every trained model from its run folder: it must reproduce its saved test predictions
        store, cfg_d = load_features(tmp / "handoff"), load_config(cfg)
        test = store.index("test")
        for name in TRAINABLE:
            model = load_run(cfg_d, name, 0, store)
            saved = pd.read_csv(tmp / "runs" / name / "seed0" / "predictions.csv").query("split == 'test'")
            assert np.allclose(predict_without(model, store, test, None, {}), saved.p_unanswerable, atol=1e-5), name
            means = input_means(store, model.needs, store.index("train"))
            for modality in ("question", "image"):
                p = predict_without(model, store, test, modality, means)
                assert p.shape == (len(test),) and np.isfinite(p).all(), (name, modality)
        try:
            load_run(cfg_d, "late", 0, store)
            raise AssertionError("late fusion has no weights and should not load")
        except FileNotFoundError:
            pass

        # 1c. Appendix A table from the run summaries
        t = table(collect([tmp / "runs"]))
        assert set(t.model) == set(TRAINABLE + ["late"]) and (t.loc[t.model == "gated_clip", "features"] == "clip").all()

        # 2. older part layout still loads and trains
        write_parts(tmp / "parts", rng)
        cfg_parts = config(tmp, tmp / "parts", tmp / "runs_parts", parts=["trainval", "test"])
        main(["--config", cfg_parts, "--model", "early", "--seeds", "0"])
        store = load_features(tmp / "parts", ["trainval", "test"])
        assert {s: len(store.index(s)) for s in ("train", "dev", "test")} == {"train": 300, "dev": 80, "test": 100}
        assert "d_q_clip" not in store.dims

        # 3. soft-label weight and the local cache
        hard = load_features(tmp / "handoff", label="hard")
        soft = load_features(tmp / "handoff", label="soft", soft_weight=0.5, cache_dir=tmp / "cache")
        share = (soft.meta["unans_votes"] / 10).clip(0, 1)
        assert np.allclose(soft.meta["target"], 0.5 * hard.meta["y"] + 0.5 * share)
        assert soft.array("clip_img").shape == (480, D_CLIP) and (tmp / "cache" / "clip_img_fp123.npy").exists()
        assert soft.dims == {"d_q": D, "d_v": D, "d_q_clip": D_CLIP, "d_v_clip": D_CLIP}
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    run()
