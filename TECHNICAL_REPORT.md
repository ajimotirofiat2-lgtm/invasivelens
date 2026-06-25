# InvasiveLens: Technical Report

## Executive Summary

InvasiveLens is a decision support system for invasive plant species identification in Portugal. It combines transfer learning on curated plant photographs with a rule-based decision engine that converts classification outputs into risk assessments and management recommendations.

The system was trained on 1,426 botanist-verified images from FloraLens across 6 species (3 invasive/counterpart pairs), supported by 1.3 million occurrence records from GBIF for geographic validation and the INVASORAS database for official invasiveness status. ResNet50 achieves 87% accuracy and EfficientNetV2-S achieves 88%, both evaluated under 5-fold spatial cross-validation to prevent geographic overfitting. McNemar's test finds no statistically significant difference between the two (p > 0.05). A baseline CNN trained from scratch reaches 82%, confirming the value of transfer learning at this data scale.

Three deployment interfaces are provided: a command-line tool for offline field use, a Flask REST API for web/mobile integration, and a Python library for programmatic access.

---

## 1. Problem Statement and Objectives

### The Problem

Invasive plant species are among the most damaging threats to biodiversity in Portugal. Species like *Arundo donax*, *Cortaderia selloana*, and *Acacia dealbata* displace native flora, degrade habitats, and impose significant economic costs. Identification currently depends on expert botanists — a slow, expensive, and poorly scalable process that creates a bottleneck for environmental monitoring teams.

### Objectives

The project set out to:

1. Build an image classifier achieving ≥85% accuracy on Portuguese invasive plant photographs using deep learning with transfer learning.
2. Implement spatial cross-validation to ensure the model generalises geographically rather than memorising location-specific features.
3. Develop a decision engine that transforms predictions into risk-tiered management recommendations.
4. Quantify prediction uncertainty and build abstention logic for low-confidence cases.
5. Compare multiple architectures (baseline CNN, ResNet50, EfficientNetV2-S) with statistical rigour using McNemar's test.
6. Deploy production-ready interfaces for real-world use.

### Key Technical Challenges

**Limited labelled data.** 1,426 images is far below what most deep learning pipelines assume. Transfer learning from ImageNet (1.3M images, 1000 classes) is essential to compensate. I verified this empirically: the from-scratch baseline achieves only 82% compared to 87–88% with pretrained weights.

**Geographic bias.** Portugal's biodiversity data is heavily concentrated in urban centres, especially Lisboa. Naïve random train/test splitting allows the model to learn location-specific cues instead of plant morphology. Spatial cross-validation with hierarchical grouping addresses this.

**Class similarity.** The species pairs were deliberately chosen for their visual similarity — these are species that non-expert observers actually confuse in the field. This makes classification harder but the system more practically useful.

**Safety-critical decisions.** Incorrectly classifying a native plant as invasive could lead to unnecessary removal. Conversely, missing an invasive species allows further spread. The confidence threshold and abstention logic provide a safety margin.

---

## 2. Technical Architecture

### System Overview

```
User Input (Photo)
    ↓
Interface Layer (CLI / REST API / Library)
    ↓
InvasiveLensPredictor (src/inference.py)
    ↓  loads checkpoint, runs forward pass, returns Prediction
Decision Engine (src/decision_engine.py)
    ↓  checks invasiveness, applies confidence thresholds
Output: Risk level + recommended action + follow-up steps
```

### Module Responsibilities

| Module | Role |
|--------|------|
| `config.py` | Centralised configuration: species pairs, hyperparameters, paths, spatial stratification settings |
| `src/cli.py` | Command-line interface with survey, identify, calibration, and report commands |
| `src/api.py` | Flask REST API with /predict, /health, /calibration, and /classes endpoints |
| `src/inference.py` | Loads trained checkpoints, runs inference, returns predictions with confidence and top-3 |
| `src/decision_engine.py` | Converts predictions to risk assessments (CRITICAL/HIGH/MEDIUM/LOW/NATIVE) with actions |
| `src/train.py` | K-fold cross-validation training loop with checkpointing and metric logging |
| `src/evaluate.py` | Accuracy, macro-F1, per-class P/R/F1, and McNemar's test implementation |
| `src/compare_models.py` | Loads fold-0 predictions from two models and runs McNemar's comparison |
| `src/data/dataset.py` | Manifest-based PyTorch Dataset that abstracts across data sources |
| `src/data/splits.py` | Hierarchical spatial grouping + StratifiedGroupKFold |
| `src/data/augmentation.py` | Training and evaluation image transforms |

### Design Decisions

**Manifest-based data abstraction.** All data sources (FloraLens, GBIF, INVASORAS) are normalised into a unified CSV format with columns: filepath, label, region, source, group, latitude, longitude. The training pipeline doesn't care where images came from — it reads the manifest. This makes adding new data sources straightforward.

**Single responsibility per module.** The inference module doesn't know about risk levels. The decision engine doesn't know about model architectures. The CLI doesn't know about training. This separation makes components independently testable and replaceable.

**Reproducibility by default.** Fixed seed (42) for all random operations, saved checkpoints per fold, saved train/test indices, all hyperparameters in config.py. Anyone can reproduce results by running the same code with the same data.

---

## 3. Data Pipeline

### Sources and Rationale

**FloraLens (1,426 images):** The primary training data. These are curated photographs from a Portuguese plant database, verified by botanists. Image quality is high and species labels are reliable. This is what the model actually learns visual features from.

**GBIF (1.3 million records):** Georeferenced occurrence records from the Global Biodiversity Information Facility. I use these primarily for geographic validation — understanding where each species occurs, building spatial groups for cross-validation, and verifying that the species pairs are ecologically sensible for Portugal.

**INVASORAS:** The official Portuguese invasive species database, maintained by the government. This provides the authoritative invasive/native classification for each species and defines the species pairs. Using an official source rather than my own classification adds credibility and avoids subjective bias.

### Data Flow

```
Raw sources → Download scripts (download_*.py)
    → Source-specific manifests
    → Merge manifests (merge_manifests.py)
    → Unified manifest CSV
    → Quality audit (quality.py)
    → Spatial grouping (splits.py)
    → StratifiedGroupKFold splits
    → ManifestDataset + augmentation
    → Training
```

### Species Pairs

| Pair | Invasive | Counterpart | Pair Confidence | Visual Challenge |
|------|----------|-------------|-----------------|-----------------|
| 1 | *Arundo donax* | *Phragmites australis* | High | Both tall riparian grasses; distinguished by stem diameter and glume hairiness |
| 2 | *Cortaderia selloana* | *Ammophila arenaria* | Medium | Both coastal grasses; confusion strongest in juvenile non-flowering plants |
| 3 | *Acacia dealbata* | *Acacia longifolia* | Medium | Both invasive Acacias in Portugal; *A. longifolia* more established, *A. dealbata* spreading faster |

A note on the third pair: *Acacia longifolia* is also classified as invasive in Portugal, though it's more naturalised than *A. dealbata*. The pair exists because field teams frequently confuse these two species, not because one is native. The `config.py` structure uses "invasive"/"native" keys for the pair format, but the notes clarify the actual status.

### Spatial Grouping Strategy

The initial approach grouped observations by Portuguese administrative district. This failed when I found that the Lisboa district contained approximately 60% of all records. Since `StratifiedGroupKFold` cannot split a single group across folds, this concentration forced at least one fold to contain ≥60% of the data, making the cross-validation meaningless.

The revised approach uses hierarchical grouping with fallback:

1. **Grid cells (~20km):** Assign each observation to a grid cell based on latitude/longitude (0.2° resolution, roughly 20km at Portugal's latitude). This is fine-grained enough that plant populations stay together but coarse enough that metro areas span multiple cells.

2. **Fallback to district:** If a grid cell has fewer than 15 observations, fall back to the district level (if the district has ≥30 observations).

3. **Fallback to macro-region:** If the district is also sparse, fall back to a macro-region (Norte-Litoral, Centro-Litoral, Sul, Ilhas).

4. **Final fallback:** "Other" for anything that doesn't fit the above.

Dense areas like metropolitan Lisbon now span many grid cells and can be properly distributed across folds. Remote areas use coarser but still geographically meaningful grouping.

### Data Augmentation

Training transforms aim to address real-world variability in field photography:

| Transform | Purpose |
|-----------|---------|
| RandomResizedCrop (scale 0.8–1.0) | Simulates different distances and framing |
| RandomHorizontalFlip | Plants have bilateral symmetry |
| RandomRotation (±15°) | Arbitrary photographing angles |
| ColorJitter (±0.2 brightness, contrast, saturation) | Variable lighting conditions |
| ImageNet normalisation | Required for pretrained model compatibility |

Validation uses only Resize + CenterCrop + normalisation — no augmentation, to get unbiased performance estimates.

---

## 4. Model Architecture and Training

### Architecture Comparison

**Baseline CNN** — A simple 4-block convolutional network (Conv2d → BatchNorm → ReLU → MaxPool, channels 3→32→64→128→256) with adaptive average pooling and dropout (0.3). Trained from scratch with random initialisation. This exists to establish a performance floor: if transfer learning doesn't meaningfully beat this, it's not worth the complexity.

**ResNet50** — A 50-layer residual network with skip connections, loaded with ImageNet pretrained weights (IMAGENET1K_V2). The original 1000-class classification head is replaced with a `Linear(2048, 6)` layer for our 6 species. All layers are fine-tuned at a low learning rate.

**EfficientNetV2-S** — Uses compound scaling to optimise depth, width, and resolution together. More parameter-efficient than ResNet50 (21M vs 25.5M) with slightly better accuracy. The classifier head `Linear(1280, 6)` replaces the original. Better suited for deployment-constrained environments (mobile, edge devices).

### Transfer Learning Approach

I fine-tune all layers rather than freezing early layers and only training the head. The rationale: our domain (Portuguese plants) is sufficiently different from ImageNet's general object categories that the early layers benefit from adaptation. Using a low learning rate (1e-4 instead of the typical 1e-3 for from-scratch training) prevents catastrophic forgetting of the pretrained features while still allowing the network to adapt.

The empirical justification is clear: 87% (ResNet50) and 88% (EfficientNetV2-S) vs 82% (from-scratch baseline). With only 1,426 training images, transfer learning provides a 5–6 percentage point improvement.

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 | Low rate for fine-tuning; prevents destroying pretrained weights |
| Batch size | 32 | Balance between memory usage and gradient stability |
| Epochs | 15 | Sufficient convergence on small dataset without overfitting |
| Optimizer | Adam | Adaptive per-parameter learning rates; works well out of the box for transfer learning |
| Loss | CrossEntropyLoss | Standard for multi-class classification with mutually exclusive classes |
| Image size | 224×224 | ImageNet standard; required for pretrained weight compatibility |
| Seed | 42 | Reproducibility across runs |

### Training Pipeline

The training script (`src/train.py`) implements the full cross-validation loop:

1. Load manifest, validate that the `group` column exists for spatial stratification.
2. Extract unique classes, build class-to-index mapping.
3. Determine optimal fold count (reduce from 5 if needed to avoid empty validation splits).
4. Create geographic splits using `StratifiedGroupKFold`.
5. For each fold: create train/val datasets with appropriate transforms, build model, train with Adam + CrossEntropyLoss, predict on validation set, compute metrics, save checkpoint.
6. For fold 0 specifically: save raw predictions, labels, confidences, and logits to `.npz` for McNemar's comparison.
7. Average metrics across folds and save summary JSON.

---

## 5. Evaluation and Results

### Cross-Validation Strategy

**5-fold cross-validation** provides more reliable performance estimates than a single train/test split, which can be misleadingly optimistic or pessimistic depending on which samples land in the test set. Each observation serves as test data exactly once across the 5 folds.

**Spatial grouping** (described in Section 3) ensures that all observations from the same geographic area end up in the same fold. This prevents the model from exploiting location-specific correlations and tests genuine generalisation to unseen regions.

**Stratification** maintains the class distribution within each fold, preventing folds with missing or severely underrepresented species.

### Results

| Model | Accuracy (mean ± std) | Macro-F1 (mean ± std) |
|-------|----------------------|----------------------|
| ResNet50 | 87% | 0.85 |
| EfficientNetV2-S | 88% | 0.86 |
| Baseline CNN | 82% | 0.80 |

Macro-F1 is reported alongside accuracy because it treats all classes equally — accuracy alone can mask poor performance on underrepresented species. The per-class precision/recall/F1 metrics (saved in the results JSON) show consistent performance across all 6 species.

### McNemar's Test

McNemar's test compares two classifiers on the same test set by building a contingency table of agreement/disagreement patterns:

```
              Model B Correct    Model B Wrong
Model A Correct    both_correct     only_A_correct
Model A Wrong      only_B_correct   both_wrong
```

The test checks whether the discordant pairs (only_A_correct, only_B_correct) are balanced. If they're significantly imbalanced, one model is genuinely better; otherwise the difference is noise.

I use the exact binomial variant rather than the chi-square approximation because the number of discordant pairs is small at this dataset scale. Both models are evaluated on fold 0 (the designated comparison set) to ensure they're tested on identical samples.

**Result:** No statistically significant difference between ResNet50 and EfficientNetV2-S (p > 0.05). The 1% accuracy gap is within the range of random variation. Either model is a reasonable choice; the decision between them comes down to deployment constraints (EfficientNetV2-S is smaller and faster, ResNet50 has broader tooling support).

### Key Findings

Transfer learning provides a consistent 5–6% improvement over training from scratch, confirming it's essential at this data scale. Spatial cross-validation is critical for geographic tasks — without it, the model could appear more accurate than it would be on truly unseen regions. The 70% confidence threshold effectively separates high-quality predictions from uncertain ones, and the per-class metrics show no single species is dramatically harder than the others.

---

## 6. Decision Engine and Risk Assessment

### Purpose

Most plant identification systems stop at classification — they output a species name and a probability. InvasiveLens goes further by running the prediction through a rule-based decision engine that assigns a risk level and recommends specific management actions. This is designed for environmental teams who need to know what to *do*, not just what the plant *is*.

### Risk Level Assignment

| Risk Level | Condition | Action | Rationale |
|------------|-----------|--------|-----------|
| **CRITICAL** | Invasive species, confidence ≥90% | Immediate removal | High certainty warrants urgent action |
| **HIGH** | Invasive species, confidence 70–90% | Containment within 30 days | Confident enough to act, but allows time for verification |
| **MEDIUM** | Any species, confidence <70% | Field verification | Too uncertain for management decisions |
| **LOW** | Non-invasive, moderate confidence | Routine monitoring | Probably not a threat |
| **NATIVE** | Native species, confidence ≥70% | No action | Document for biodiversity records |

### Decision Logic

```
Prediction from model
    → Is confidence < 70%?
        YES → MEDIUM risk, recommend field verification
        NO  → Is species invasive (per INVASORAS)?
            NO  → NATIVE risk, no action
            YES → Is confidence ≥ 90%?
                YES → CRITICAL risk, immediate removal
                NO  → HIGH risk, containment
```

### Confidence Thresholds

The default 70% threshold was chosen to balance two failure modes:
- **False positives** (removing native plants) are expensive and ecologically harmful.
- **False negatives** (missing invasive species) allow further spread and increase future removal costs.

The threshold is configurable per use case. A more conservative deployment might use 85% (fewer false positives, more manual reviews). A more aggressive one might use 60% (catches more invasives but generates more false alarms).

The 90% threshold for CRITICAL risk adds an extra safety layer — immediate removal is the most drastic action, so it requires the highest confidence.

### Uncertainty Quantification

The model's softmax output provides a probability distribution over all classes. The confidence score is the maximum probability: how much of the total probability mass the model assigns to its top prediction. Below the threshold, the system abstains and defers to human judgement.

This is a deliberate design choice. In a safety-critical application like environmental management, a confidently wrong prediction is worse than admitting uncertainty. The calibration data (saved in fold-0 predictions) can be used to verify that confidence scores are reasonably well-calibrated — that 90% confident predictions are actually correct approximately 90% of the time.

---

## 7. Deployment

### CLI (`src/cli.py`)

The primary interface for field teams. Supports single-image assessment, batch processing of survey directories, CSV export, and report generation. Works fully offline once model checkpoints are downloaded — important for remote fieldwork without connectivity.

Key commands:
```bash
python -m src.cli survey --image photo.jpg        # Single image
python -m src.cli survey --batch folder/           # Batch
python -m src.cli calibration --model resnet50      # Model calibration info
python -m src.cli report --results results.csv      # Management report
```

### REST API (`src/api.py`)

A Flask application for web and mobile integration. The model is loaded at startup, and predictions are served via a POST endpoint. Includes health check, calibration info, and class listing endpoints.

```bash
python -m src.api --port 5000
curl -X POST http://localhost:5000/predict -F "file=@image.jpg"
```

### Python Library

The `InvasiveLensPredictor` and `InvasiveSpeciesDecisionEngine` classes can be imported directly into custom Python workflows for programmatic access.

### Performance

Inference takes approximately 1 second per image on CPU, faster on GPU. A batch of 50 images completes in about 2 minutes including report generation. Model checkpoints are 25MB (ResNet50) or 21MB (EfficientNetV2-S).

---

## 8. Limitations and Future Directions

### Current Limitations

**Species coverage:** The system handles 3 species pairs (6 species total). This is a proof of concept — the framework is designed to scale, but expanding coverage requires additional training data and validation.

**Dataset size:** 1,426 images is on the small side for deep learning. Transfer learning compensates effectively, but more data — especially difficult edge cases (partial views, juvenile plants, mixed stands) — would improve robustness.

**Visual features only:** The model uses only pixel-level information from photographs. Contextual signals like habitat type, elevation, season, geographic coordinates, and surrounding vegetation aren't incorporated. A multi-modal approach combining visual and contextual features is a natural extension.

**Heuristic thresholds:** The 70% and 90% confidence thresholds were set based on judgement rather than formal optimisation. A cost-benefit analysis quantifying the real-world costs of false positives vs false negatives could produce better-calibrated thresholds.

**No production deployment:** The system has been evaluated against held-out validation data but hasn't been deployed with an environmental agency. Real-world conditions (variable camera hardware, weather, lighting, growth stages) may reveal gaps not captured by the current dataset.

### Future Directions

- Expand to more species pairs, prioritising those with highest ecological impact and identification difficulty.
- Integrate satellite imagery for landscape-scale invasion detection, complementing ground-level photography.
- Explore multi-modal approaches that combine image features with contextual metadata.
- Experiment with advanced augmentation strategies (mixup, cutout, AutoAugment) to improve generalisation.
- Conduct formal threshold optimisation using decision-theoretic methods.
- Deploy with a Portuguese environmental agency to collect feedback and iterate.
- Develop a mobile application with on-device inference for fully offline field use.

---

## 9. Conclusion

InvasiveLens demonstrates that transfer learning can achieve practical accuracy levels (87–88%) for invasive plant identification with limited training data, provided that evaluation methodology accounts for geographic structure. The spatial cross-validation approach, which evolved through a failed first attempt with district-level grouping, is arguably the most important methodological contribution — it prevents the inflated accuracy estimates that random splitting produces for geographically structured data.

Beyond the classification itself, the decision engine and uncertainty quantification move the system from a research prototype toward a practical tool. Environmental teams need actionable outputs, not just species names and probabilities. The tiered risk system, confidence-based abstention, and structured follow-up recommendations are designed to integrate with existing field survey workflows rather than replace expert judgement entirely.

The framework is extensible: new species pairs can be added through the config and manifest system, new model architectures slot in through the build_model factory, and the decision engine's thresholds and risk levels are configurable. The three deployment interfaces (CLI, API, library) cover the range from field technicians with laptops to developers building web applications.

All objectives were met: accuracy exceeds the 85% target, spatial cross-validation prevents geographic overfitting, McNemar's test provides statistical rigour for model comparison, and the system is deployable in multiple modes. The primary areas for improvement are data scale and species coverage — both solvable with additional data collection and straightforward retraining.
