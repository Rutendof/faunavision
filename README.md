# FaunaVision

Fine-grained recognition of lookalike Southern African animal species, with
explanations of *why* the model chose one species over its nearest lookalike.

> Work in progress. This README grows with the project.

## Approach

- **Confusion clusters.** Species are chosen in small groups of genuine
  lookalikes (same genus or family), so the benchmark actually tests
  fine-grained recognition.
- **Contrastive explanations.** "Why species A and not B?" shown with
  difference-of-logits Grad-CAM and patch matching against reference images.
- **Location and season prior.** Uses where and when the photo was taken.
- **Honest uncertainty.** Calibrated confidence, a fallback to genus or family
  when unsure, and an out-of-distribution test on camera-trap images.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pytest
```

## Step 1: choose species

```bash
python scripts/select_species.py --place-id 113055 --min-observations 300
```

Queries the public iNaturalist API (throttled to about 1 request per second,
cached under `data/cache/`) and writes a proposal to `reports/species_selection/`.

## Step 2: camera-trap dataset

Uses LILA's Snapshot Safari 2024 (CDLA-Permissive 1.0) from 12 Southern
African reserves, plus LILA's pre-computed MegaDetector boxes.

```bash
# metadata only (~120 MB download), into data/raw/lila/
python scripts/build_camtrap_index.py                          # ~6 min, streams 3 GB of JSON
python scripts/make_manifest.py --test-projects PLN,KRU,CDB    # splits + reports/data/coverage.md
python scripts/download_crops.py --limit 20                    # smoke test; full run on Kaggle
```

Test reserves (Pilanesberg, Kruger, Camdeboo) are never seen in training, so
results show how the model transfers to a new conservancy.

## Step 3: frozen DINOv2 baseline (model 1)

Run `notebooks/01_kaggle_crops_and_dinov2_baseline.ipynb` on a free Kaggle
GPU. Locally, the same scripts run on a small sample:

```bash
python scripts/extract_features.py --crops data/crops_mini --manifest data/processed/manifest_mini.csv \
    --model vit_small_patch14_dinov2.lvd142m --out data/features_mini
python scripts/linear_probe.py --features data/features_mini --run smoke
```

Reports land in `reports/results/<run>/` with top-1/5, macro-F1, calibration,
per-class recall and lookalike-pair accuracy (e.g. white vs black rhino).

## Data and licensing

Only research-grade observations with Creative Commons licences are used.
Images are downloaded locally for training and are **not** redistributed with
this repository. Attribution is kept in the dataset metadata.
