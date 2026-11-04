"""Evaluate every trained run and write the tables and figures for the report.

Run from the repo root in Colab after src.features and src.train:
    python -m src.evaluate --config configs/default.yaml --ablation-config configs/ablation.yaml \
        --out /content/drive/MyDrive/vizwiz_individual/results --cache-dir /content/features_local

Thresholds and the late-fusion weight come from the development split; the test split is only scored.
Methods: percentile bootstrap CIs (Efron & Tibshirani 1993); exact McNemar tests (McNemar 1947; Dietterich 1998)
with Holm (1979) correction; expected calibration error (Guo et al. 2017); Brier score (Brier 1950); AUPRC for
imbalanced classes (Saito & Rehmsmeier 2015); Mann-Whitney U test (Mann & Whitney 1947).
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import binomtest, mannwhitneyu, spearmanr  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from src.data import load_features  # noqa: E402
from src.load import input_means, load_run, predict_without  # noqa: E402
from src.train import load_config  # noqa: E402

MAIN = ["majority", "photo_quality", "question_only", "image_only", "late", "early", "cross_attention", "gated",
        "clip_early", "gated_clip"]
FLOORS = ["majority", "photo_quality"]
UNIMODAL = ["question_only", "image_only"]
FUSIONS = ["late", "early", "cross_attention", "gated", "clip_early", "gated_clip"]
NAMES = {"question_only": "Question-only", "image_only": "Image-only", "late": "Late fusion",
         "early": "Early fusion", "cross_attention": "Cross-attention", "gated": "Gated fusion",
         "clip_early": "CLIP early fusion", "gated_clip": "CLIP gated fusion",
         "gated_md0": "Gated, no modality dropout", "gated_md30": "Gated, modality dropout 0.3",
         "gated_clip_md0": "CLIP gated, no modality dropout", "cross_attention_l2": "Cross-attention, 2 blocks",
         "majority": "Majority class", "photo_quality": "Photo quality only"}
KEY = ["image_only", "early", "cross_attention", "gated", "clip_early", "gated_clip"]
T_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


# ---------------------------------------------------------------- metrics
def macro_f1(y, yhat):
    tp = np.sum((yhat == 1) & (y == 1)); fp = np.sum((yhat == 1) & (y == 0)); fn = np.sum((yhat == 0) & (y == 1))
    tn = len(y) - tp - fp - fn
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / max(2 * tn + fn + fp, 1)
    return (f1_pos + f1_neg) / 2


def tune_threshold(y, p):
    return float(T_GRID[int(np.argmax([macro_f1(y, (p >= t).astype(int)) for t in T_GRID]))])


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def metrics(y, p, t):
    yhat = (p >= t).astype(int)
    tp = np.sum((yhat == 1) & (y == 1)); fn = np.sum((yhat == 0) & (y == 1))
    tn = np.sum((yhat == 0) & (y == 0)); fp = np.sum((yhat == 1) & (y == 0))
    return {"macro_f1": macro_f1(y, yhat), "f1_unans": 2 * tp / max(2 * tp + fp + fn, 1),
            "recall_unans": tp / max(tp + fn, 1), "precision_unans": tp / max(tp + fp, 1),
            "balanced_acc": (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2,
            "auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p)}


# ---------------------------------------------------------------- loading
def read_runs(runs_dir, meta):
    """model -> seed -> {"dev": p, "test": p, "gate": gate or None, "summary": dict, "history": DataFrame}."""
    order = {s: meta.loc[meta.split == s, "pair_id"].values for s in ("dev", "test")}
    runs = {}
    for f in sorted(Path(runs_dir).glob("*/seed*/predictions.csv")):
        pred = pd.read_csv(f)
        model, seed = f.parent.parent.name, int(f.parent.name.replace("seed", ""))
        r = {}
        for s in ("dev", "test"):
            sub = pred[pred.split == s].set_index("pair_id")
            if not set(order[s]) <= set(sub.index):
                r = None
                break
            r[s] = sub.loc[order[s], "p_unanswerable"].values
            if s == "test":
                r["gate"] = sub.loc[order[s], "gate_q"].values if "gate_q" in sub else None
        if r is None:
            print(f"skipped {f}: it does not cover every dev and test pair")
            continue
        summ, hist = f.parent / "summary.json", f.parent / "history.csv"
        r["summary"] = json.loads(summ.read_text()) if summ.exists() else {}
        r["history"] = pd.read_csv(hist) if hist.exists() else None
        runs.setdefault(model, {})[seed] = r
    return runs


def floor_models(meta):
    """Reference floors, no fusion: always the training prevalence, and a logistic regression on photo
    sharpness and brightness only (how far photo quality alone gets)."""
    from sklearn.linear_model import LogisticRegression
    tr = meta[meta.split == "train"]
    feats = lambda d: np.c_[np.log10(d.sharpness.fillna(d.sharpness.median()) + 1), d.brightness.fillna(128) / 255]
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(feats(tr), tr.y.astype(int))
    runs = {}
    for name in FLOORS:
        r = {"gate": None, "summary": {"params": 0 if name == "majority" else 3}, "history": None}
        for s in ("dev", "test"):
            d = meta[meta.split == s]
            r[s] = np.full(len(d), tr.y.mean()) if name == "majority" else lr.predict_proba(feats(d))[:, 1]
        runs[name] = {0: r}
    return runs


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- sections
def data_quality(meta, manifest, tab, fig_dir):
    out = {}
    for s in ("train", "dev", "test"):
        m = meta[meta.split == s]
        out[s] = {"pairs": int(len(m)), "unanswerable_rate": round(float(m.y.mean()), 3)}
    tr = meta[meta.split == "train"]
    lab = meta[meta.split != "test"]
    out["question_words_median"] = float(meta.q_words.median())
    out["question_words_p95"] = float(meta.q_words.quantile(0.95))
    out["repeated_question_share"] = round(float(meta.question.str.lower().duplicated(keep=False).mean()), 3)
    out["rare_question_share_test"] = round(float(meta.loc[meta.split == "test", "rare_question"].mean()), 3)
    out["ambiguous_share"] = round(float(((lab.y == 0) & (lab.unans_votes >= 3)).mean()), 3)
    s0, s1 = tr.loc[tr.y == 0, "sharpness"].dropna(), tr.loc[tr.y == 1, "sharpness"].dropna()
    out["sharpness_median"] = {"answerable": round(float(s0.median()), 1), "unanswerable": round(float(s1.median()), 1)}
    out["sharpness_mannwhitney_p"] = float(mannwhitneyu(s0, s1).pvalue)
    out["unreadable_photos"] = manifest.get("timing", {}).get("images", {}).get("unreadable")
    out["truncated_questions"] = manifest.get("timing", {}).get("questions", {}).get("truncated_share")
    top = (meta.assign(q=meta.question.str.lower()).groupby("q")["y"].agg(count="size", unanswerable_rate="mean")
           .sort_values("count", ascending=False).head(10).round(3))
    top.to_csv(tab / "top_questions.csv")
    sb = (meta[meta.split == "test"].groupby("sharp_bin")["y"].agg(pairs="size", unanswerable_rate="mean").round(3))
    sb.to_csv(tab / "test_unanswerable_by_sharpness.csv")
    pd.Series({k: v for k, v in out.items() if not isinstance(v, dict)}).to_csv(tab / "data_quality.csv")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    counts = meta.groupby(["split", "y"]).size().unstack().reindex(["train", "dev", "test"])
    counts.columns = ["Answerable", "Unanswerable"]
    counts.plot(kind="bar", stacked=True, ax=ax[0], rot=0, color=["#4C72B0", "#DD8452"])
    ax[0].set(title="(a) Class balance by split", xlabel="Split", ylabel="Image-question pairs")
    for yv, c, lbl in ((0, "#4C72B0", "Answerable"), (1, "#DD8452", "Unanswerable")):
        ax[1].hist(meta.loc[meta.y == yv, "q_words"].clip(upper=40), bins=40, alpha=0.6, density=True, color=c, label=lbl)
    ax[1].set(title="(b) Question length", xlabel="Words (clipped at 40)", ylabel="Density")
    ax[1].legend(frameon=False)
    ax[2].boxplot([np.log10(s0 + 1), np.log10(s1 + 1)], showfliers=False)
    ax[2].set_xticks([1, 2], ["Answerable", "Unanswerable"])
    ax[2].set(title="(c) Photo sharpness (train)", ylabel="log10(variance of Laplacian + 1)")
    save(fig, fig_dir / "data_quality.png")
    return out


def seed_results(runs, meta, models):
    y_dev, y_test = meta.loc[meta.split == "dev", "y"].values, meta.loc[meta.split == "test", "y"].values
    rows, thr = [], {}
    for m in models:
        for seed, r in sorted(runs.get(m, {}).items()):
            t = tune_threshold(y_dev, r["dev"])
            thr[(m, seed)] = t
            rows.append({"model": m, "seed": seed, "threshold": t, **metrics(y_test, r["test"], t)})
    per_seed = pd.DataFrame(rows)
    summary = per_seed.drop(columns=["seed", "threshold"]).groupby("model").agg(["mean", "std"])
    summary = summary.reindex([m for m in models if m in summary.index])
    return per_seed, summary, thr


def bootstrap(runs, meta, thr, models, seed=0, n_boot=1000):
    y = meta.loc[meta.split == "test", "y"].values.astype(int)
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    boots, rows = {}, []
    for m in models:
        if seed not in runs.get(m, {}):
            continue
        p = runs[m][seed]["test"]
        yhat = (p >= thr[(m, seed)]).astype(int)
        f1 = np.array([macro_f1(y[b], yhat[b]) for b in idx])
        ap = np.array([average_precision_score(y[b], p[b]) for b in idx])
        boots[m] = {"macro_f1": f1, "auprc": ap, "correct": yhat == y}
        rows.append({"model": m, "macro_f1": macro_f1(y, yhat), "macro_f1_lo": np.percentile(f1, 2.5),
                     "macro_f1_hi": np.percentile(f1, 97.5), "auprc": average_precision_score(y, p),
                     "auprc_lo": np.percentile(ap, 2.5), "auprc_hi": np.percentile(ap, 97.5)})
    return pd.DataFrame(rows).set_index("model"), boots


def fusion_gains(boots, ref):
    rows = []
    for m in FUSIONS:
        if m in boots and ref in boots:
            for k in ("macro_f1", "auprc"):
                d = boots[m][k] - boots[ref][k]
                rows.append({"fusion": m, "vs": ref, "metric": k, "gain": float(np.mean(d)),
                             "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
                             "ci_excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)})
    return pd.DataFrame(rows)


def mcnemar_table(boots, models):
    rows = []
    ms = [m for m in models if m in boots]
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            ca, cb = boots[a]["correct"], boots[b]["correct"]
            n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            rows.append({"model_a": a, "model_b": b, "a_right_b_wrong": n01, "a_wrong_b_right": n10, "p": p})
    t = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    k = len(t)
    t["p_holm"] = np.minimum(1, np.maximum.accumulate([(k - i) * p for i, p in enumerate(t.p)]))
    t["significant"] = t.p_holm < 0.05
    return t


def curves(runs, meta, fig_dir, seed=0):
    y = meta.loc[meta.split == "test", "y"].values.astype(int)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for m in KEY:
        if seed not in runs.get(m, {}):
            continue
        p = runs[m][seed]["test"]
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, label=NAMES[m])
        ax[1].plot(rc, pr, label=NAMES[m])
        edges = np.linspace(0, 1, 11)
        b = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        xs = [p[b == i].mean() for i in range(10) if (b == i).sum() >= 20]
        ys = [y[b == i].mean() for i in range(10) if (b == i).sum() >= 20]
        ax[2].plot(xs, ys, marker="o", ms=3, label=NAMES[m])
    ax[0].plot([0, 1], [0, 1], ls=":", c="grey")
    ax[1].axhline(y.mean(), ls=":", c="grey")
    ax[2].plot([0, 1], [0, 1], ls=":", c="grey")
    ax[0].set(title="(a) ROC, test", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set(title="(b) Precision-recall, test", xlabel="Recall (unanswerable)", ylabel="Precision")
    ax[2].set(title="(c) Reliability, test", xlabel="Predicted probability", ylabel="Observed unanswerable rate")
    ax[2].legend(frameon=False, fontsize=7)
    save(fig, fig_dir / "roc_pr_calibration.png")


def comparison_figure(boot_table, fig_dir):
    t = boot_table.reindex([m for m in MAIN if m in boot_table.index])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colours = ["#BBBBBB" if m in FLOORS else "#8C8C8C" if m in UNIMODAL else "#55A868" if m.startswith(("clip", "gated_clip")) else "#4C72B0" for m in t.index]
    xerr = [np.clip(t.macro_f1 - t.macro_f1_lo, 0, None), np.clip(t.macro_f1_hi - t.macro_f1, 0, None)]
    ax.barh([NAMES[m] for m in t.index], t.macro_f1, color=colours, xerr=xerr, capsize=3)
    ax.invert_yaxis()
    ax.set(xlabel="Macro-F1 on test (95% bootstrap CI, seed 0)", xlim=(max(0, t.macro_f1_lo.min() - 0.05), 1),
           title="Floors and single modality (grey), ViT/BERT fusion (blue), CLIP fusion (green)")
    save(fig, fig_dir / "model_comparison.png")


def subgroups(runs, meta, thr, models, tab, fig_dir):
    test = meta[meta.split == "test"].reset_index(drop=True)
    rows = []
    for col in ("sharp_bin", "rare_question"):
        for g, sub in test.groupby(col):
            ix = sub.index.values
            y = sub.y.values.astype(int)
            for m in models:
                for seed, r in runs.get(m, {}).items():
                    p = r["test"][ix]
                    rows.append({"by": col, "group": str(g), "model": m, "seed": seed, "pairs": len(ix),
                                 "unanswerable_rate": y.mean(), "macro_f1": macro_f1(y, (p >= thr[(m, seed)]).astype(int)),
                                 "auprc": average_precision_score(y, p)})
    t = (pd.DataFrame(rows).groupby(["by", "group", "model"])
         .agg(pairs=("pairs", "first"), unanswerable_rate=("unanswerable_rate", "first"),
              macro_f1=("macro_f1", "mean"), auprc=("auprc", "mean")).round(3).reset_index())
    t.to_csv(tab / "subgroups.csv", index=False)
    sb = t[(t.by == "sharp_bin") & t.model.isin(KEY)].pivot(index="model", columns="group", values="macro_f1")
    sb = sb.reindex([m for m in KEY if m in sb.index])[[c for c in ("blurry", "medium", "sharp") if c in sb.columns]]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    sb.rename(index=NAMES).plot(kind="bar", ax=ax, rot=20, color=["#C44E52", "#DD8452", "#4C72B0"])
    ax.set(ylabel="Macro-F1 (mean of 3 seeds)", xlabel="", title="Test macro-F1 by photo sharpness",
           ylim=(max(0, sb.values.min() - 0.1), min(1, sb.values.max() + 0.05)))
    ax.legend(title="Photo", frameon=False)
    save(fig, fig_dir / "subgroups_sharpness.png")
    return t


def gate_analysis(runs, meta, tab):
    test = meta[meta.split == "test"].reset_index(drop=True)
    rows, corr = [], {}
    for m in ("gated", "gated_clip"):
        gates = [r["gate"] for r in runs.get(m, {}).values() if r["gate"] is not None]
        if not gates:
            continue
        g = np.mean(gates, axis=0)
        d = test.assign(gate_q=g)
        tbl = d.groupby(["sharp_bin", "y"]).gate_q.mean().unstack()
        for sb_, row in tbl.iterrows():
            rows.append({"model": m, "sharp_bin": sb_, "answerable": row.get(0), "unanswerable": row.get(1)})
        corr[m] = {"spearman_gate_vs_sharpness": float(spearmanr(d.gate_q, d.sharpness, nan_policy="omit")[0]),
                   "gate_mean": float(g.mean()), "gate_sd_across_pairs": float(g.std())}
    pd.DataFrame(rows).round(3).to_csv(tab / "gate_by_sharpness.csv", index=False)
    return corr


def learning_curves(runs, fig_dir, seed=0):
    ms = [m for m in MAIN if seed in runs.get(m, {}) and runs[m][seed]["history"] is not None]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4), sharey=True)
    for ax, m in zip(axes.flat, ms):
        h = runs[m][seed]["history"]
        ax.plot(h.epoch, h.train_loss, marker="o", ms=2.5, label="Train (eval mode)")
        ax.plot(h.epoch, h.dev_loss, marker="o", ms=2.5, label="Development")
        ax.set(title=NAMES[m], xlabel="Epoch")
    for ax in axes.flat[len(ms):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("Log loss")
    axes[1, 0].set_ylabel("Log loss")
    axes[0, 0].legend(frameon=False)
    save(fig, fig_dir / "learning_curves.png")
    gaps = {m: round(float(runs[m][seed]["history"].dev_loss.iloc[-1] - runs[m][seed]["history"].train_loss.iloc[-1]), 3) for m in ms}
    return gaps


def modality_ablation(cfg, store, meta, device, tab, fig_dir, seed=0):
    test, train = store.index("test"), store.index("train")
    y = meta.loc[meta.split == "test", "y"].values.astype(int)
    rows = []
    for m in ["early", "cross_attention", "gated", "clip_early", "gated_clip"]:
        try:
            model = load_run(cfg, m, seed, store, device)
        except (FileNotFoundError, KeyError) as e:
            print(f"ablation skipped for {m}: {e}")
            continue
        means = input_means(store, model.needs, train)
        for mod in (None, "question", "image"):
            p = predict_without(model, store, test, mod, means, device)
            rows.append({"model": m, "input": {None: "both", "question": "question replaced by mean",
                                               "image": "image replaced by mean"}[mod],
                         "auprc": average_precision_score(y, p), "auroc": roc_auc_score(y, p)})
    t = pd.DataFrame(rows).round(3)
    t.to_csv(tab / "modality_ablation.csv", index=False)
    if len(t):
        piv = t.pivot(index="model", columns="input", values="auprc")
        piv = piv.reindex([m for m in KEY if m in piv.index])
        fig, ax = plt.subplots(figsize=(7, 3.4))
        piv.rename(index=NAMES)[["both", "question replaced by mean", "image replaced by mean"]].plot(
            kind="bar", ax=ax, rot=20, color=["#4C72B0", "#DD8452", "#8C8C8C"])
        ax.axhline(y.mean(), ls=":", c="grey", label="Chance (prevalence)")
        ax.set(ylabel="Test AUPRC (seed 0)", xlabel="", title="Modality ablation: one input replaced by its training mean")
        ax.legend(frameon=False, fontsize=7)
        save(fig, fig_dir / "modality_ablation.png")
    return t


def latency(cfg, store, runs, meta, device, seed=0, reps=200):
    """Median milliseconds per single query: fusion head on cached features, plus each encoder family."""
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    def bench(fn, n):
        for _ in range(10):
            fn()
        sync()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            sync()
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))

    q = store.index("test")[:1]
    out = {}
    for m in [m for m in MAIN if m not in FLOORS + ["late"]]:
        try:
            model = load_run(cfg, m, seed, store, device)
        except (FileNotFoundError, KeyError):
            continue
        batch = store.batch(q, model.needs, device)
        with torch.no_grad():
            out[m] = {"head_ms": bench(lambda: model(batch), reps)}
    if {"question_only", "image_only"} <= out.keys():
        out["late"] = {"head_ms": out["question_only"]["head_ms"] + out["image_only"]["head_ms"]}
    path = meta.loc[meta.split == "test", "path"].iloc[0] if "path" in meta else None
    enc = {}
    if path and Path(path).exists():
        try:
            from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, CLIPModel
            from src.features import CLIP_MEAN, CLIP_STD, IMAGE_MODEL, MAX_Q, TEXT_MODEL, CLIP_MODEL, Photos, norm
            question = [meta.loc[meta.split == "test", "question"].iloc[0]]
            proc = AutoImageProcessor.from_pretrained(IMAGE_MODEL)
            vit = AutoModel.from_pretrained(IMAGE_MODEL).to(device).eval()
            bert, tok = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval(), AutoTokenizer.from_pretrained(TEXT_MODEL)
            clip, ctok = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval(), AutoTokenizer.from_pretrained(CLIP_MODEL)
            photo = Photos([path])

            @torch.no_grad()
            def vit_bert():
                x = photo[0][0].unsqueeze(0).to(device)             # includes JPEG decode and resize
                vit(pixel_values=norm(x, proc.image_mean, proc.image_std))
                bert(**tok(question, padding="max_length", truncation=True, max_length=MAX_Q, return_tensors="pt").to(device))

            @torch.no_grad()
            def clip_enc():
                x = photo[0][0].unsqueeze(0).to(device)
                clip.vision_model(pixel_values=norm(x, CLIP_MEAN, CLIP_STD))
                clip.text_model(**ctok(question, padding=True, return_tensors="pt").to(device))

            enc = {"vit_bert_ms": bench(vit_bert, 30), "clip_ms": bench(clip_enc, 30)}
        except Exception as e:                                   # encoders unavailable: report heads only
            print(f"encoder timing skipped: {e}")
    for m, v in out.items():
        family = "clip_ms" if m.startswith(("clip", "gated_clip")) else "vit_bert_ms"
        if family in enc:
            v["end_to_end_ms"] = v["head_ms"] + enc[family]
    return out, enc


def failure_cases(runs, meta, thr, model, tab, fig_dir, seed=0, k=6):
    test = meta[meta.split == "test"].reset_index(drop=True)
    p = runs[model][seed]["test"]
    t = thr[(model, seed)]
    d = test.assign(p=p, pred=(p >= t).astype(int))
    missed = d[(d.y == 1) & (d.pred == 0)].nsmallest(k, "p").assign(error="missed unanswerable")
    false_alarm = d[(d.y == 0) & (d.pred == 1)].nlargest(k, "p").assign(error="false alarm")
    cases = pd.concat([missed, false_alarm])
    cols = ["pair_id", "question", "y", "p", "sharp_bin", "q_freq_train", "unans_votes", "error"]
    cases[[c for c in cols if c in cases]].round(3).to_csv(tab / "failure_cases.csv", index=False)
    err = d[d.pred != d.y]
    summary = {"model": model, "threshold": t, "errors": int(len(err)),
               "error_rate_by_sharpness": d.assign(e=d.pred != d.y).groupby("sharp_bin").e.mean().round(3).to_dict(),
               "missed_unanswerable": int(((d.y == 1) & (d.pred == 0)).sum()),
               "false_alarms": int(((d.y == 0) & (d.pred == 1)).sum())}
    if "path" in cases and cases.path.map(lambda x: isinstance(x, str) and Path(x).exists()).all():
        from PIL import Image, ImageOps
        fig, axes = plt.subplots(2, k, figsize=(2.2 * k, 5.2))
        for ax, (_, r) in zip(axes.flat, cases.iterrows()):
            with Image.open(r.path) as im:
                ax.imshow(ImageOps.exif_transpose(im).convert("RGB").resize((160, 160)))
            q = (r.question[:34] + "...") if len(r.question) > 37 else r.question
            ax.set_title(f"{q}\np={r.p:.2f}, {r.sharp_bin}", fontsize=6.5)
            ax.axis("off")
        axes[0, 0].text(-0.15, 0.5, "Missed\nunanswerable", transform=axes[0, 0].transAxes, ha="right", va="center")
        axes[1, 0].text(-0.15, 0.5, "False\nalarm", transform=axes[1, 0].transAxes, ha="right", va="center")
        fig.suptitle(f"Failure cases, {NAMES[model]} (photos: VizWiz, CC BY 4.0)", y=1.0)
        save(fig, fig_dir / "failure_cases.png")
    return summary


def efficiency(runs, lat, models):
    rows = []
    for m in models:
        sums = [r["summary"] for r in runs.get(m, {}).values()]
        rows.append({"model": m, "params": sums[0].get("params") if sums else None,
                     "sec_per_epoch": np.mean([s.get("sec_per_epoch", np.nan) for s in sums]) if sums else None,
                     "epochs": np.mean([s.get("epochs_run", np.nan) for s in sums]) if sums else None,
                     "head_ms": lat.get(m, {}).get("head_ms"), "end_to_end_ms": lat.get(m, {}).get("end_to_end_ms")})
    return pd.DataFrame(rows).set_index("model").round(3)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate all runs and write report tables and figures")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ablation-config", default="configs/ablation.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--skip-models", action="store_true", help="skip modality ablation and latency")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(args.out)
    tab, fig_dir = out / "tables", out / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feat = Path(cfg["paths"]["features_dir"])
    store = load_features(feat, cfg["data"].get("parts"), "hard", 1.0, args.cache_dir)
    meta = store.meta.copy()
    manifest = json.loads((feat / "manifest.json").read_text()) if (feat / "manifest.json").exists() else {}
    report = {"data": data_quality(meta, manifest, tab, fig_dir), "extraction_timing": manifest.get("timing")}

    runs = {**read_runs(cfg["paths"]["runs_dir"], meta), **floor_models(meta)}
    models = [m for m in MAIN if m in runs]
    per_seed, summary, thr = seed_results(runs, meta, models)
    per_seed.round(4).to_csv(tab / "test_metrics_per_seed.csv", index=False)
    summary.round(3).to_csv(tab / "test_metrics_summary.csv")
    boot_table, boots = bootstrap(runs, meta, thr, models)
    boot_table.round(3).to_csv(tab / "test_bootstrap_ci_seed0.csv")
    dev_auprc = {m: np.mean([average_precision_score(meta.loc[meta.split == "dev", "y"], r["dev"]) for r in runs[m].values()])
                 for m in UNIMODAL if m in runs}
    ref = max(dev_auprc, key=dev_auprc.get)                      # best single modality, chosen on dev
    gains = fusion_gains(boots, ref)
    gains.round(4).to_csv(tab / "fusion_gain_vs_best_unimodal.csv", index=False)
    mcn = mcnemar_table(boots, models)
    mcn.round(4).to_csv(tab / "mcnemar_holm.csv", index=False)
    comparison_figure(boot_table, fig_dir)
    curves(runs, meta, fig_dir)
    sub = subgroups(runs, meta, thr, models, tab, fig_dir)
    gate = gate_analysis(runs, meta, tab)
    gaps = learning_curves(runs, fig_dir)
    best = max([m for m in models if m not in FLOORS], key=lambda m: np.mean([average_precision_score(meta.loc[meta.split == "dev", "y"], r["dev"])
                                              for r in runs[m].values()]))
    fails = failure_cases(runs, meta, thr, best, tab, fig_dir)

    abl = None
    if args.ablation_config and Path(args.ablation_config).exists():
        acfg = load_config(args.ablation_config)
        aruns = read_runs(acfg["paths"]["runs_dir"], meta)
        if aruns:
            names = list(acfg["models"])
            pairs = {"gated_md0": "gated", "gated_md30": "gated", "gated_clip_md0": "gated_clip", "cross_attention_l2": "cross_attention"}
            both = {**{m: runs[m] for m in set(pairs.values()) if m in runs}, **aruns}
            order = list(dict.fromkeys(m for n in names for m in (pairs.get(n), n) if m in both))
            _, abl, _ = seed_results(both, meta, order)
            abl.round(3).to_csv(tab / "ablation.csv")

    lat, enc, mod_abl = {}, {}, None
    if not args.skip_models:
        mod_abl = modality_ablation(cfg, store, meta, device, tab, fig_dir)
        lat, enc = latency(cfg, store, runs, meta, device)
    eff = efficiency(runs, lat, models)
    eff.to_csv(tab / "efficiency.csv")

    mean = summary.xs("mean", axis=1, level=1)
    report.update({
        "test_summary_mean": mean.round(3).to_dict(orient="index"),
        "test_summary_std": summary.xs("std", axis=1, level=1).round(3).to_dict(orient="index"),
        "best_single_modality": ref, "best_model_by_dev": best,
        "fusion_gain": gains.round(4).to_dict(orient="records"),
        "targets": {m: {"macro_f1>=0.78": bool(mean.loc[m, "macro_f1"] >= 0.78), "auprc>=0.75": bool(mean.loc[m, "auprc"] >= 0.75)}
                    for m in FUSIONS if m in mean.index},
        "gate": gate, "overfit_gap_at_last_epoch": gaps, "failure_cases": fails,
        "encoder_ms": enc, "efficiency": eff.reset_index().to_dict(orient="records"),
        "modality_ablation": None if mod_abl is None else mod_abl.to_dict(orient="records"),
        "ablation": None if abl is None else abl.xs("mean", axis=1, level=1).round(3).to_dict(orient="index"),
        "significant_pairs": int(mcn.significant.sum()), "pairs_tested": int(len(mcn))})
    (out / "summary.json").write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    print("\n=== Test results (mean +/- std over seeds) ===")
    show = summary[["macro_f1", "auprc", "auroc", "ece"]].round(3)
    print(show.rename(index=NAMES).to_string())
    print(f"\nBest single modality (dev): {NAMES[ref]}. Fusion gains (seed 0, paired bootstrap):")
    g = gains[gains.metric == "macro_f1"] if len(gains) else gains
    for _, r in g.iterrows():
        print(f"  {NAMES[r.fusion]:<20} macro-F1 {r.gain:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
    print(f"\nMcNemar (Holm): {int(mcn.significant.sum())} of {len(mcn)} model pairs differ significantly")
    print(f"Gate: {json.dumps(gate)}")
    if mod_abl is not None and len(mod_abl):
        print("\nModality ablation (test AUPRC, seed 0):")
        print(mod_abl.pivot(index="model", columns="input", values="auprc").rename(index=NAMES).to_string())
    if len(eff):
        print("\nEfficiency:")
        print(eff.rename(index=NAMES).to_string())
    if abl is not None:
        print("\nAblation (test, mean over seeds):")
        print(abl[["macro_f1", "auprc"]].round(3).rename(index=NAMES).to_string())
    print(f"\nTables: {tab}\nFigures: {fig_dir}\nSummary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
