# Flagging Unanswerable Visual Questions with Image-Text Fusion

**Vishnu Vardhan Manikandan** - Multimodal Data Fusion, Adelaide University, Semester 2 2026 (Project 2.2, individual submission)

Blind and low-vision people photograph objects and ask spoken questions about them, but about 28% of those questions cannot be answered from the photo because it is blurred, mis-framed or misses the object (Gurari et al. 2018). This project predicts whether a question is answerable from the **photo and question together**, and compares early, late, cross-attention and gated fusion, on two encoder families (ViT/BERT and CLIP), against single-modality baselines.

## Reproduce everything

Open [`notebooks/02_pipeline.ipynb`](notebooks/02_pipeline.ipynb) in Google Colab, choose Runtime > Change runtime type > **T4 GPU**, then Runtime > **Run all** and allow Google Drive access. One run (about 75 minutes) downloads the data, extracts the features, trains every model with 3 seeds, runs the ablation and writes all tables and figures to `MyDrive/vizwiz_individual/results/`.

The same steps from a terminal at the repo root:

```
python -m tests.test_smoke                                        # quick check (CPU, about a minute)
python -m src.features --data-dir /content/data --out /content/features
python -m src.train --config configs/default.yaml --model all
python -m src.train --config configs/ablation.yaml --model gated_md0   # and gated_md30, gated_clip_md0, cross_attention_l2
python -m src.evaluate --config configs/default.yaml --ablation-config configs/ablation.yaml --out results
python scripts/config_table.py --out results/tables/appendix_a_configs.csv
```

## Results

Official test split (8,000 pairs), mean (standard deviation) over three seeds; thresholds tuned on the development split. Full tables and figures are in [`results/`](results/) (aggregate numbers only).

| Model | Encoders | Macro-F1 | AUPRC | ECE |
|---|---|---|---|---|
| Majority class | - | 0.423 | 0.268 | 0.001 |
| Photo quality only | sharpness, brightness | 0.617 | 0.455 | 0.214 |
| Question-only | BERT | 0.663 (0.001) | 0.536 (0.005) | 0.167 |
| Image-only | ViT | 0.694 (0.001) | 0.597 (0.003) | 0.150 |
| Late fusion | ViT + BERT | 0.742 (0.004) | 0.674 (0.002) | 0.165 |
| Early fusion | ViT + BERT | 0.771 (0.004) | 0.733 (0.002) | 0.127 |
| Cross-attention | ViT + BERT | 0.766 (0.004) | 0.719 (0.004) | 0.112 |
| Gated fusion | ViT + BERT | 0.756 (0.001) | 0.708 (0.004) | 0.105 |
| CLIP early fusion (reference) | CLIP | 0.800 (0.002) | 0.780 (0.002) | 0.106 |
| CLIP gated fusion (reference) | CLIP | 0.793 (0.003) | 0.763 (0.003) | 0.116 |

Key findings:

- Every fusion model beats both single modalities; early fusion gains +0.072 macro-F1 over image-only (95% bootstrap interval +0.060 to +0.083).
- The encoder matters more than the fusion design: CLIP adds 0.029 macro-F1 to early fusion, while the ViT/BERT fusion strategies differ by at most 0.015.
- The gate in gated fusion barely varies across pairs, and modality dropout (0, 0.15, 0.3) or a second cross-attention block changes macro-F1 by at most 0.002.
- Every model answers within 35 ms per query on a T4 GPU, mostly encoder time.
- Errors concentrate on blurry photos (18.6% misclassified versus 12.8% for sharp ones), the main bias discussed in the report.

## Repository structure

```
configs/default.yaml              main settings: paths, training, every model
configs/ablation.yaml             design ablation: modality dropout, cross-attention depth
src/features.py                   download, labels, photo quality, ViT / BERT / CLIP features
src/data.py                       loads the features and assigns train / dev / test
src/models.py                     baselines, early fusion, cross-attention, gated fusion
src/train.py                      shared training loop (python -m src.train)
src/load.py                       reload trained models; modality ablation helpers
src/evaluate.py                   all report tables and figures (python -m src.evaluate)
scripts/config_table.py           Appendix A configuration table
tests/test_smoke.py               training-level check on synthetic data
tests/test_pipeline.py            features + training + evaluation check with stub encoders
docs/data_contract.md             feature, prediction and run-folder formats
notebooks/01_vizwiz_pilot.ipynb   pilot analysis from the group proposal (Assignment 2.1)
notebooks/02_pipeline.ipynb       the full individual pipeline, saved with its outputs
results/                          final test results: tables, figures, summary.json
results/figures/pilot_*.png       pilot figures from the proposal
```

## Data

VizWiz-VQA (Gurari et al. 2018), released under a Creative Commons Attribution 4.0 International licence (VizWiz n.d.). The data is **not stored in this repository**; `src/features.py` downloads it from the official source. Splits: official train (20,523 pairs) for training, official validation (4,319) for model selection and thresholds, official test (8,000, labels released April 2026) for final scoring only.

## Ethics

Only code and aggregate results are shared: no images, questions or annotations are redistributed. Example photos in the report and presentation are attributed to VizWiz (CC BY 4.0).

## Licence

Code: MIT (see `LICENSE`). Data: Creative Commons Attribution 4.0, owned by the VizWiz team (VizWiz n.d.).

## References

Gurari, D, Li, Q, Stangl, AJ, Guo, A, Lin, C, Grauman, K, Luo, J & Bigham, JP 2018, 'VizWiz grand challenge: answering visual questions from blind people', in *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*, IEEE, Salt Lake City, USA, pp. 3608-3617.

VizWiz n.d., *Visual question answering*, VizWiz, viewed 20 October 2026, <https://vizwiz.org/tasks-and-datasets/vqa/>.

Full reference lists are in the notebooks.
