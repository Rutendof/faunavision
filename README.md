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

Optional extras: `.[export]` (ONNX export, step 7) and `.[api]` (inference
API, step 8). Training (steps 3 to 7) also needs PyTorch and timm, which
Kaggle images already include.

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

## Step 4: detector triage

```bash
python scripts/triage_eval.py --projects PLN,KRU,CDB
```

Stage 1 of the pipeline: MegaDetector removes empty frames before any species
model runs. Scored per sequence (one trigger burst), because rangers review
events, not frames. For each confidence threshold it reports empty sequences
removed (work saved) and animal sequences kept, overall and for priority
species. MegaDetector saw some Snapshot Safari data in training, so treat
these numbers as optimistic.

## Step 5: fine-tuned models (models 2 to 5)

On Kaggle, `notebooks/02_kaggle_models_distill_export.ipynb` runs steps 5 and 7
one stage per session (`kaggle kernels push -p kaggle/02_models`).

```bash
# Model 2: ResNet-50 baseline
python scripts/train.py --model resnet50.a1_in1k --run resnet50 --epochs 15
# Main model: DINOv2 ViT-B/14 with layer-wise LR decay
python scripts/train.py --model vit_base_patch14_dinov2.lvd142m --run dinov2_b14_ft --lr 5e-4 --layer-decay 0.7
# Model 5: refine the main model at 448 px for 2-3 epochs
python scripts/train.py --model vit_base_patch14_dinov2.lvd142m --run dinov2_b14_448     --img-size 448 --epochs 3 --init checkpoints/dinov2_b14_ft/best.pt
```

AdamW with warmup and cosine decay, label smoothing, RandAugment,
class-balanced sampling (inverse square-root frequency) and weight EMA. A
checkpoint is written every epoch, and `--resume` continues an interrupted
Kaggle session. The best epoch by validation macro-F1 is calibrated
(temperature fitted on validation) and scored on the unseen test reserves.
Results go to `reports/results/<run>/`, including `val_logits.npz` and
`test_logits.npz`.

## Step 6: contrastive explanations

`faunavision.explain.contrastive_gradcam(model, x, class_a, class_b)` targets
`logit[A] - logit[B]`, so the heatmap shows what favours white rhino **over**
black rhino rather than just where the rhino is. It works for CNNs and ViTs.
`overlay(image, cam)` draws the map on the crop. The maps are diagnostic aids,
not proof that the model reasons like an expert.

## Step 7: distilled student and ONNX export (model 6)

```bash
# Distil model 5 into a small student (the teacher scores each augmented batch)
python scripts/train.py --model efficientnet_b0.ra_in1k --run student_b0     --teacher checkpoints/dinov2_b14_448/best.pt --kd-alpha 0.7 --kd-temp 2
# Export to a single .onnx file and check it against PyTorch
python scripts/export_onnx.py --ckpt checkpoints/student_b0/best.pt --out exports/model.onnx     --metrics reports/results/student_b0/metrics.json
```

The export writes `model.json` next to the `.onnx`, holding the class list,
input size, normalisation, calibration temperature and the lookalike-review
margin. That margin is tuned on validation logits as the smallest gap between
the top two predictions that flags `--review-recall` (default 0.8) of
lookalike errors. The script prints the recall it reaches and the share of
crops it sends to review: the ranger workload that recall costs. Errors whose
runner-up is outside the cluster can never be flagged, so the target may be
out of reach for some models.

## Step 8: inference API

```bash
pip install -e ".[api]"
FAUNAVISION_MODEL=exports/model.onnx uvicorn faunavision.api:app --app-dir src
```

ONNX Runtime on CPU, with no PyTorch needed at serving time.

- `GET /health`: loaded model, input size and class count.
- `POST /predict` (multipart `file`, optional `top_k`): one animal crop in,
  top-k species out with common name, probability, cluster and priority, plus
  `priority_alert` and `lookalike_review` flags for triage.

Uploads are capped at 10 MB, limited to JPEG, PNG or WebP, and checked
against decompression bombs.

### Docker / Hugging Face Spaces

```bash
docker build -t faunavision-api .        # needs exports/model.onnx and exports/model.json
docker run --rm -p 7860:7860 faunavision-api
```

The image holds only the API code and the model, with no PyTorch and no
training data (`.dockerignore` whitelists what goes in). It follows the
Spaces conventions: port 7860 and uid 1000. To deploy, push the Dockerfile,
`pyproject.toml`, `src/` and the model (via Git LFS) to a Docker Space whose
README front matter has `sdk: docker` and `app_port: 7860`.

## Data and licensing

Only research-grade observations with Creative Commons licences are used.
Images are downloaded locally for training and are **not** redistributed with
this repository. Attribution is kept in the dataset metadata.
