"""Appendix A: one row per model configuration, built from the saved run summaries (all seeds).

Run in Colab from the repo root, with Google Drive mounted:
    python scripts/config_table.py
It prints the table and saves it as results/appendix_a_configs.csv in the results folder.
Training settings shared by every model (AdamW, learning rate 0.001, weight decay 0.01, batch 256,
up to 30 epochs, early stopping on dev AUPRC with patience 4) live in configs/default.yaml.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

TEAM = "/content/drive/MyDrive/vizwiz_individual"
SHOWN = ["hidden", "d_model", "heads", "layers", "dropout", "modality_dropout"]


def settings(hp):
    return ", ".join(f"{k}={hp[k]}" for k in SHOWN if k in hp)


def collect(runs_dirs):
    rows = []
    for d in runs_dirs:
        for f in sorted(Path(d).glob("*/seed*/summary.json")):
            s = json.loads(f.read_text())
            hp = s.get("hyperparameters") or {}
            name = s.get("model", f.parent.parent.name)
            rows.append({"set": Path(d).name, "model": name, "seed": s.get("seed"),
                         "class": hp.get("base", name), "features": hp.get("features", "vit_bert") if hp else "-",
                         "settings": settings(hp), "params": s.get("params"), "epochs_run": s.get("epochs_run"),
                         "sec_per_epoch": s.get("sec_per_epoch"), "peak_vram_gb": s.get("peak_vram_gb"),
                         "dev_auprc": s.get("best_dev_auprc")})
    if not rows:
        raise SystemExit(f"no summary.json found under {runs_dirs}")
    return pd.DataFrame(rows)


def table(runs):
    t = runs.groupby(["set", "model"], sort=False).agg(
        cls=("class", "first"), features=("features", "first"), settings=("settings", "first"),
        params=("params", "first"), seeds=("seed", "nunique"), epochs=("epochs_run", "mean"),
        sec_per_epoch=("sec_per_epoch", "mean"), peak_vram_gb=("peak_vram_gb", "max"),
        dev_auprc_mean=("dev_auprc", "mean"), dev_auprc_std=("dev_auprc", "std")).reset_index()
    t = t.rename(columns={"cls": "class"})
    t["params"] = t["params"].astype("Int64")
    return t.round({"epochs": 1, "sec_per_epoch": 2, "peak_vram_gb": 3, "dev_auprc_mean": 3, "dev_auprc_std": 3})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the Appendix A configuration table")
    ap.add_argument("--runs-dirs", nargs="*", default=[f"{TEAM}/runs", f"{TEAM}/runs_ablation"])
    ap.add_argument("--out", default=f"{TEAM}/results/appendix_a_configs.csv")
    args = ap.parse_args(argv)
    t = table(collect([d for d in args.runs_dirs if Path(d).is_dir()]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.to_string(index=False))
    print(f"\nSaved {args.out}")
    return t


if __name__ == "__main__":
    main()
