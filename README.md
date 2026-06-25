# InvasiveLens

An image classification pipeline for invasive plant species identification and management in Portugal.

InvasiveLens takes a photograph of a plant, identifies the species using a fine-tuned deep learning model, and produces a risk assessment with management recommendations. It goes beyond classification — the decision engine assigns risk levels (CRITICAL, HIGH, MEDIUM, LOW, NATIVE) and outputs specific actions like "contain within 30 days" or "schedule field verification."

---

## Quick Demo

```bash
python -m src.cli survey --image data_raw/floralens/Phragmites_australis/Phragmites_australis_1.jpg
```

Output:
```
MEDIUM RISK: Arundo donax (confidence: 27.3%)
Action: Monitoring
Reasoning: Low confidence detection (27.3%). Requires field verification.
```

The model flagged this as a possible *Arundo donax* but only at 27.3% confidence, so it recommended manual verification instead of triggering a removal action. That's the uncertainty quantification at work — the system refuses to make high-stakes decisions when it isn't confident enough.

---

## Why This Exists

Invasive plants are a serious problem in Portugal. Species like *Arundo donax*, *Cortaderia selloana*, and *Acacia dealbata* outcompete native flora, degrade habitats, and cost significant resources to remove once established. The bottleneck is identification: it typically requires an expert botanist, which means delays, high costs, and limited coverage.

I built InvasiveLens to make that identification step faster and more accessible. A field team can photograph suspicious plants during a survey, run them through the model, and get prioritised results with confidence scores — all without waiting for an expert. The system is deliberately conservative: if it's unsure, it says so.

---

## How It Works

The pipeline has three stages:

1. **Classification** — A ResNet50 model (fine-tuned from ImageNet) processes the input image and outputs a probability distribution across 6 species classes.

2. **Uncertainty check** — If the top-class probability is below 70%, the system abstains from making a management decision and flags the image for manual review.

3. **Decision engine** — For confident predictions, the engine looks up whether the identified species is invasive (using pairs defined in `config.py`) and assigns a risk level based on confidence:
   - ≥90% confidence on an invasive species → **CRITICAL** → immediate removal
   - 70–90% confidence on an invasive species → **HIGH** → containment within 30 days
   - <70% confidence → **MEDIUM** → field verification needed
   - Native species → **NATIVE** → no action

The full flow from CLI input to output:

```
python -m src.cli survey --image photo.jpg
    → cli.py parses args, creates DecisionEngine
    → DecisionEngine calls InvasiveLensPredictor
    → Predictor loads checkpoint, runs inference, returns Prediction
    → DecisionEngine checks invasiveness + confidence → assigns risk
    → cli.py prints the result
```

---

## Usage

**Single image:**
```bash
python -m src.cli survey --image your_photo.jpg
```

**Batch processing (whole folder):**
```bash
python -m src.cli survey --batch survey_photos/ --export results.csv
```

**REST API for web/mobile integration:**
```bash
python -m src.api --port 5000
```
Then POST images to `http://localhost:5000/predict`.

**Quick species ID (no decision engine, just classification):**
```bash
python -m src.cli identify your_photo.jpg
```

---

## Data Sources

All training data comes from real-world sources — no synthetic generation.

| Source | What It Provides | Scale |
|--------|-----------------|-------|
| **FloraLens** | Curated, botanist-verified plant photographs from Portugal | 1,426 images |
| **GBIF** | Georeferenced occurrence records from the Global Biodiversity Information Facility | 1.3M records |
| **INVASORAS** | Official Portuguese government invasive species database | Species status + pair definitions |

The 1,426 images are split across 6 species in 3 pairs. Each pair consists of one invasive species and one visually similar counterpart that field teams commonly confuse:

| Invasive | Counterpart | Why This Pair |
|----------|-------------|---------------|
| *Arundo donax* | *Phragmites australis* | Both tall riparian grasses, distinguished by stem diameter and glume hairiness |
| *Cortaderia selloana* | *Ammophila arenaria* | Coastal grasses with similar morphology, especially in juvenile non-flowering stage |
| *Acacia dealbata* | *Acacia longifolia* | Both invasive Acacias in Portugal; *A. longifolia* is more established while *A. dealbata* spreads more aggressively |

Note on the Acacia pair: *A. longifolia* is also classified as invasive in Portugal but is more naturalised. The pair exists because field teams frequently confuse these two Acacias, not because one is native.

---

## Models

I trained three architectures to compare transfer learning against a from-scratch baseline:

| Model | Accuracy | Macro-F1 | Parameters | Notes |
|-------|----------|----------|------------|-------|
| **ResNet50** | 87% | 0.85 | 25.5M | Recommended for production. Fine-tuned from ImageNet. |
| **EfficientNetV2-S** | 88% | 0.86 | 21M | Slightly better accuracy, fewer params. Good for mobile. |
| **Baseline CNN** | 82% | 0.80 | Small | 4 conv blocks from scratch. Establishes the performance floor. |

The 5–6% gap between the baseline and transfer learning models justifies the added complexity. McNemar's test on the fold-0 held-out set shows no statistically significant difference between ResNet50 and EfficientNetV2-S (p > 0.05), so the 1% gap is likely noise.

Transfer learning is essential here because 1,426 images isn't enough to train a deep network from scratch without severe overfitting. Loading ImageNet weights gives the model strong low-level features (edges, textures, shapes) that transfer well to plant identification. I fine-tune all layers at a low learning rate (1e-4) to adapt without destroying the pretrained representations.

---

## Spatial Cross-Validation

I don't use random train/test splits. Random splitting causes spatial leakage — if photos from the same location appear in both training and test sets, the model can learn location-specific cues (lighting conditions, soil colour, camera style) instead of actual plant features.

My first attempt at fixing this was to group by Portuguese administrative district. That fell apart when I discovered that the Lisboa district alone contained ~60% of the records. Since group-based k-fold can't split a single group across folds, one fold ended up being 60%+ of the data, which made the validation meaningless.

The solution is a hierarchical spatial grouping with fallback:

1. Assign each observation to a ~20km grid cell based on lat/lon
2. If the grid cell has ≥15 observations → use grid cell as group
3. Otherwise, if the district has ≥30 observations → use district
4. Otherwise → fall back to macro-region (Norte/Centro/Sul/Ilhas) or "Other"

This way, dense areas like the Lisbon metro span many grid cells and can be distributed across folds, while sparse regions get coarser but still spatially meaningful grouping. The implementation uses `StratifiedGroupKFold` from scikit-learn — stratified to maintain class balance, grouped to prevent leakage.

---

## Project Structure

```
invasivelens/
├── src/
│   ├── cli.py                  # Command-line interface
│   ├── api.py                  # Flask REST API
│   ├── inference.py            # Model loading + prediction
│   ├── decision_engine.py      # Risk assessment logic
│   ├── train.py                # K-fold training pipeline
│   ├── evaluate.py             # Metrics + McNemar's test
│   ├── compare_models.py       # Head-to-head model comparison
│   ├── data/
│   │   ├── dataset.py          # Unified manifest-based Dataset
│   │   ├── splits.py           # Spatial cross-validation
│   │   ├── augmentation.py     # Train/eval transforms
│   │   ├── quality.py          # Data auditing
│   │   └── download_*.py       # Per-source downloaders
│   └── models/
│       ├── baseline_cnn.py     # 4-block CNN from scratch
│       └── transfer_models.py  # ResNet50 + EfficientNetV2-S wrappers
├── config.py                   # All hyperparameters, paths, species pairs
├── checkpoints/                # Saved model weights per fold
├── data_raw/                   # Raw training images
├── manifests/                  # Unified CSV manifests
└── results/                    # Metrics JSON + fold-0 predictions
```

---

## Installation

```bash
pip install -e ".[dev]"
```

## Training

```bash
python -m src.train --manifest manifests/combined_manifest.csv --model resnet50 --epochs 15
```

## Model Comparison (McNemar's Test)

```bash
python -m src.compare_models --model-a resnet50 --model-b efficientnet_v2_s
```

## Running Tests

```bash
pytest
```

---

## Limitations

- Coverage is limited to 3 species pairs (6 species). The framework scales, but I haven't expanded beyond this proof of concept.
- 1,426 training images is small by deep learning standards. Transfer learning compensates, but more data would help — especially difficult edge cases.
- The system only uses visual features. Contextual signals like habitat type, elevation, season, or geographic coordinates aren't incorporated into predictions.
- The 70% confidence threshold was chosen heuristically. A proper cost-benefit analysis (false positive removal cost vs. false negative ecological damage) could produce a better-calibrated threshold.
- No production deployment yet. Validation metrics may not fully reflect real-world performance under varied conditions.

---

## Further Reading

- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) — Full project report with methodology, results, and analysis
- [config.py](config.py) — All configuration: species pairs, hyperparameters, spatial stratification settings
