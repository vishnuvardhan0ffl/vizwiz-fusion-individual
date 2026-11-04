# Data and output formats (Project 2.2)

The formats that `src/features.py`, `src/train.py`, `src/load.py` and `src/evaluate.py` read and write.

## Feature hand-off

`python -m src.features` writes one set of features for all pairs to `features_dir` (see `configs/default.yaml`):

```
features/
    manifest.json        every array: file name, shape, dtype, encoder
    meta.csv             one row per pair, in the same order as every array
    <name>.npy
```

`meta.csv` required columns (`src/data.py` checks them):

| Column | Meaning |
|---|---|
| pair_id | image file name, unique (e.g. VizWiz_test_00000001.jpg) |
| question | question text |
| official_split | train, val or test (mapped to train, dev, test) |
| y | 1 = unanswerable, 0 = answerable (1 - answerable flag); test labels come from VQA_test.json |
| unans_votes | how many of the 10 crowd answers are "unanswerable" or "unsuitable" (soft labels only) |

Extra columns for the analysis: `sharpness` (variance of the Laplacian), `brightness`, `sharp_bin` (blurry / medium / sharp, cut at training tertiles), `q_freq_train`, `rare_question` (question text seen at most once in training), `q_words`, `answer_type` and `path` (photo file, for the failure-case gallery). `meta.csv` contains the questions, so it stays on the Colab disk and is never committed.

Arrays (row i belongs to row i of `meta.csv`):

| Name | Shape | Type | Content | Batch key |
|---|---|---|---|---|
| img_cls | (N, 768) | float16 | ViT-B/16 [CLS] vector | v_vec |
| img_tok | (N, 49, 768) | float16 | ViT patch tokens, 14x14 average-pooled to 7x7 | v_tok |
| img_ok | (N,) | uint8 | 1 = photo readable | - |
| txt_cls | (N, 768) | float16 | BERT [CLS] vector | - |
| txt_mean | (N, 768) | float16 | BERT mask-aware mean of token states | q_vec |
| txt_tok | (N, 40, 768) | float16 | BERT token states, padded or truncated to 40 | q_tok |
| txt_mask | (N, 40) | uint8 | BERT attention mask | q_mask |
| clip_img | (N, 512) | float16 | CLIP ViT-B/32 image embedding | v_clip |
| clip_txt | (N, 512) | float16 | CLIP ViT-B/32 text embedding | q_clip |

Encoders: `google/vit-base-patch16-224-in21k`, `bert-base-uncased`, `openai/clip-vit-base-patch32`; each photo is decoded once (EXIF rotation, 224x224 resize without cropping) and fed to both image encoders; CLIP embeddings are L2-normalised. Arrays are loaded only when a model needs them.


## Models

| Name | Class (`base`) | Inputs (`features`) |
|---|---|---|
| question_only | QuestionOnly | BERT mean |
| image_only | ImageOnly | ViT [CLS] |
| early | EarlyFusion | BERT mean + ViT [CLS] |
| late | (train.py) | question_only and image_only predictions, weight tuned on dev |
| clip_early | EarlyFusion | CLIP text + image (reference) |
| cross_attention | CrossAttentionFusion | BERT tokens + ViT region tokens |
| gated | GatedFusion | BERT mean + ViT [CLS] |
| gated_clip | GatedFusion | CLIP text + image |

Floor models, computed inside `src/evaluate.py`: `majority` (training prevalence) and `photo_quality` (logistic regression on sharpness and brightness).

## Run outputs

`python -m src.train` writes one folder per model and seed: `runs/<model>/seed<k>/`

| File | Content |
|---|---|
| model.pt | best weights, selected on dev AUPRC |
| history.csv | epoch, train_loss, dev_loss, dev_auprc, seconds |
| predictions.csv | pair_id, split, model, seed, y, p_unanswerable (+ gate_q for gated models) for dev and test |
| summary.json | settings, parameter count, epochs run, best epoch, seconds per epoch, peak GPU memory |

## Loading a trained model

`src/load.py` rebuilds any trained model from `runs/<model>/seed<k>/` (hyperparameters from its `summary.json`, weights from `model.pt`), so evaluation, latency timing and ablations need no retraining:

```python
from src.train import load_config
from src.data import load_features
from src.load import load_run, input_means, predict_without
cfg = load_config("configs/default.yaml")
store = load_features(cfg["paths"]["features_dir"])
model = load_run(cfg, "gated", seed=0, store=store, device="cuda")      # runs_dir=... for runs_ablation
test = store.index("test")
means = input_means(store, model.needs, store.index("train"))
p_full = predict_without(model, store, test, None, means, device="cuda")
p_no_image = predict_without(model, store, test, "image", means, device="cuda")
```

`predict_without` replaces the question or image inputs with their training-set mean (modality ablation). For cross-attention the question mask is kept, so question length still reaches the model.

Appendix A (one row per configuration, from all `summary.json` files): `python scripts/config_table.py`.

## Adding a model

1. New architecture: add a class to `src/models.py` with a `needs` tuple and the constructor `(d_q, d_v, **hyperparameters)`, and register it in `MODELS`.
2. New variant of an existing class: just add a config entry with `base:` and `features:`.
3. Add the name to `TRAINABLE` in `src/train.py`.

## Rules

- Never use `split == "test"` for tuning, thresholds or early stopping. Tune on dev only.
- Never use answers, answer_type or unans_votes as model inputs: they decide the label.
- Every reported number must come from a config file and a seed in `runs/`.
- `python -m tests.test_smoke` and `python -m tests.test_pipeline` must pass after any code change.
