# InvasiveLens — Project Report

**AI-Based Invasive Plant Identification and Management Support for Portugal**

---

## 1. Introduction

Invasive plant species are among the most serious threats to Portugal's ecosystems. Species such as *Arundo donax*, *Cortaderia selloana*, and *Acacia dealbata* displace native flora, degrade habitats, and impose substantial economic costs on environmental management. Identifying these species accurately in the field remains a bottleneck — it requires trained botanists, which limits the speed and scale at which environmental agencies can respond.

InvasiveLens addresses this problem by combining deep learning image classification with a rule-based decision engine. Given a photograph of a plant, the system identifies the species, checks whether it is classified as invasive in Portugal, assesses prediction confidence, and outputs a risk-tiered management recommendation. The system is designed to augment expert judgement rather than replace it: when the model is uncertain, it says so and defers to human review.

The project focuses on three species pairs selected for their ecological significance and visual similarity — these are species that non-experts routinely confuse in the field. The system achieves 87–88% accuracy under spatially-aware cross-validation, and is deployed as a command-line tool, REST API, and Python library.

---

## 2. Data

Three authoritative, real-world data sources were used:

| Source | Content | Scale | Role |
|--------|---------|-------|------|
| **FloraLens** | Botanist-verified Portuguese plant photographs | 1,426 images | Primary training data |
| **GBIF** | Georeferenced occurrence records | 1.3M records | Geographic validation and spatial grouping |
| **INVASORAS** | Official Portuguese invasive species classifications | Species-level | Authoritative invasive/native status |

**FloraLens** provides the images the model learns from. All labels are verified by botanists, which ensures high label quality. **GBIF** provides geographic distribution data used to build the spatial cross-validation groups. **INVASORAS** is the official government database that defines which species are invasive — using an authoritative source avoids subjective classification.

### Species Pairs

| Invasive Species | Counterpart | Visual Challenge |
|-----------------|-------------|-----------------|
| *Arundo donax* | *Phragmites australis* | Both tall riparian grasses; distinguished by stem diameter and glume hairiness |
| *Cortaderia selloana* | *Ammophila arenaria* | Both coastal grasses; confusion strongest in juvenile non-flowering plants |
| *Acacia dealbata* | *Acacia longifolia* | Both invasive Acacias in Portugal; *A. longifolia* is more established while *A. dealbata* is spreading faster |

Note: both Acacias are invasive in Portugal. This pair exists because field teams frequently confuse them, not because one is native.

---

## 3. Data Organisation

All data sources are normalised into a unified CSV manifest format with columns: `filepath`, `label`, `region`, `source`, `group`, `latitude`, `longitude`. The training pipeline reads only the manifest — it does not need to know which source an image came from.

### Data Pipeline

```
Raw sources → Download scripts (src/data/download_*.py)
    → Source-specific manifests
    → Merge (src/data/merge_manifests.py)
    → Unified manifest CSV
    → Quality audit (src/data/quality.py)
    → Spatial grouping (src/data/splits.py)
    → K-fold splits → Training
```

### Spatial Grouping

Observations are assigned to geographic groups for cross-validation using a hierarchical strategy:

1. **Grid cells (~20 km)** — each observation is placed in a 0.2° grid cell based on its coordinates. Dense areas like Lisbon span many cells and can be distributed across folds.
2. **District fallback** — if a grid cell has fewer than 15 observations, it falls back to the administrative district (if the district has ≥30 observations).
3. **Macro-region fallback** — if the district is also sparse, it falls back to a macro-region (Norte-Litoral, Centro-Litoral, Sul, Ilhas).

This approach was developed after an initial attempt using district-level grouping failed — the Lisboa district contained approximately 60% of all records, making fold-based splitting impossible since a single group cannot be split across folds.

### Data Augmentation

Training images undergo random resized cropping (scale 0.8–1.0), horizontal flipping, rotation (±15°), and colour jitter (±0.2 brightness/contrast/saturation) to simulate field photography variability. Validation images use only resize, centre crop, and ImageNet normalisation.

---

## 4. Methods

### Models

Three architectures were compared:

- **Baseline CNN** — A 4-block convolutional network (Conv2d → BatchNorm → ReLU → MaxPool, channels 3→32→64→128→256) trained from scratch. Establishes a performance floor.
- **ResNet50** — 50-layer residual network, initialised with ImageNet pretrained weights. Classification head replaced with Linear(2048, 6). All layers fine-tuned at learning rate 1e-4.
- **EfficientNetV2-S** — Compound-scaled architecture, more parameter-efficient than ResNet50 (21M vs 25.5M parameters). Classification head replaced with Linear(1280, 6).

### Transfer Learning

All layers are fine-tuned rather than frozen, using a learning rate of 1e-4 (10× lower than typical from-scratch training). This allows the network to adapt to plant-specific features without destroying the general visual features learned from ImageNet's 1.3 million images.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Batch size | 32 |
| Epochs | 15 |
| Optimiser | Adam |
| Loss function | CrossEntropyLoss |
| Image size | 224 × 224 |
| Random seed | 42 |

### Evaluation Strategy

**5-fold spatial cross-validation** using `StratifiedGroupKFold`. Stratification maintains class balance within each fold. Geographic grouping ensures all observations from the same area are in the same fold, preventing the model from exploiting location-specific features (lighting, soil colour, camera characteristics) rather than learning plant morphology.

### Statistical Comparison

**McNemar's test** (exact binomial variant) compares the two transfer learning models on identical test samples (fold 0). The test checks whether the pattern of disagreement between the two models is symmetric or skewed — if symmetric, neither model is significantly better.

### Decision Engine

The decision engine converts classification outputs into management recommendations:

| Condition | Risk Level | Action |
|-----------|-----------|--------|
| Confidence < 70% (any species) | MEDIUM | Field verification |
| Non-invasive species, confidence ≥ 70% | NATIVE | No action |
| Invasive species, confidence 70–90% | HIGH | Containment within 30 days |
| Invasive species, confidence ≥ 90% | CRITICAL | Immediate removal |

The 70% threshold balances false positives (unnecessary removal of native plants) against false negatives (missing invasive species). Both thresholds are configurable.

---

## 5. Results

### Classification Performance

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| ResNet50 | 87% | 0.85 |
| EfficientNetV2-S | 88% | 0.86 |
| Baseline CNN | 82% | 0.80 |

Transfer learning provides a 5–6 percentage point improvement over the from-scratch baseline, confirming its value at this data scale. Per-class metrics show consistent performance across all 6 species.

### Statistical Comparison

McNemar's test finds no statistically significant difference between ResNet50 and EfficientNetV2-S (p > 0.05). The 1% accuracy gap falls within random variation. Either model is a valid choice; the selection depends on deployment constraints.

---

## 6. Analysis

### Transfer Learning Effectiveness

The 5–6% accuracy gap between pretrained models and the from-scratch baseline demonstrates that transfer learning is not optional at this data scale. With only 1,426 training images, a randomly initialised network lacks sufficient data to learn robust visual features. ImageNet pretraining provides a strong feature foundation that fine-tuning adapts to our domain.

### Spatial Cross-Validation Impact

The hierarchical spatial grouping strategy prevents inflated accuracy estimates. Without geographic separation, the model could achieve higher apparent accuracy by learning location-specific correlations rather than plant morphology — performance that would not generalise to new survey areas. The failed district-level grouping attempt (where Lisboa's 60% data share made fold splitting impossible) motivated the grid-cell approach, which distributes dense areas across multiple folds while keeping sparse areas in meaningful geographic clusters.

### Uncertainty Quantification

The confidence-based abstention mechanism is critical for practical deployment. At the 70% threshold, predictions below this level are routed for manual review rather than acted upon. This is a deliberate safety trade-off: in environmental management, a confidently wrong decision (removing a native plant, or ignoring an invasive one) has real ecological and economic costs. The calibration data saved from fold-0 predictions can verify that confidence scores are well-calibrated.

### Explainability

Grad-CAM visualisations confirm that the model attends to plant morphology (leaf shape, stem structure, flower heads) rather than background features. This was validated on both ResNet50 and EfficientNetV2-S using the `src/explain/gradcam.py` implementation.

---

## 7. Deployment

Three deployment interfaces serve different use cases:

### Command-Line Interface

```bash
python -m src.cli survey --image photo.jpg           # Single image
python -m src.cli survey --batch folder/ --export results.csv  # Batch
```

Works fully offline once checkpoints are downloaded. Designed for field teams with laptops.

### REST API

```bash
python -m src.api --port 5000
curl -X POST http://localhost:5000/predict -F "file=@image.jpg"
```

Flask application for web and mobile integration. Returns JSON with species, confidence, risk level, and recommended action.

### Python Library

```python
from src.inference import InvasiveLensPredictor
from src.decision_engine import InvasiveSpeciesDecisionEngine

predictor = InvasiveLensPredictor(model_name="resnet50")
engine = InvasiveSpeciesDecisionEngine()
```

For researchers and developers building custom workflows.

### Inference Performance

Approximately 1 second per image on CPU. A batch of 50 field survey photos processes in roughly 2 minutes including report generation.

---

## 8. References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

2. Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. *ICML 2021*.

3. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.

4. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.

5. GBIF.org. Global Biodiversity Information Facility. https://www.gbif.org

6. Flora-On / FloraLens. Interactive Flora of Portugal. https://flora-on.pt

7. INVASORAS. Invasive Plants in Portugal. http://invasoras.pt

8. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR 2009*.

9. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

10. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*.

---

## 9. Contributions

**System Design and Architecture** — Designed the end-to-end pipeline from data ingestion to deployment, including the manifest-based data abstraction, modular code architecture, and three deployment interfaces.

**Data Pipeline** — Built download, import, and merge scripts for three data sources (FloraLens, GBIF, INVASORAS). Implemented quality auditing and data versioning.

**Spatial Cross-Validation** — Developed the hierarchical spatial grouping strategy (grid cells → district → macro-region) after identifying and solving the dominant-district problem in the initial district-level approach.

**Model Training and Evaluation** — Implemented the k-fold training loop with spatial stratification, per-fold checkpointing, and McNemar's statistical comparison. Trained and evaluated three architectures.

**Decision Engine** — Designed and implemented the confidence-based risk assessment system that converts classifications into actionable management recommendations with uncertainty quantification.

**Explainability** — Implemented Grad-CAM from scratch (no external dependency) for both ResNet50 and EfficientNetV2-S architectures to verify model attention patterns.

**Testing** — Built test suite covering spatial splitting (regression tests for the Lisboa dominance bug), GradCAM localisation, inference, decision engine logic, and data quality.

---

## Appendix A: Project Structure

```
invasivelens/
├── src/
│   ├── cli.py                  # Command-line interface
│   ├── api.py                  # REST API (Flask)
│   ├── inference.py            # Model loading and prediction
│   ├── decision_engine.py      # Risk assessment logic
│   ├── train.py                # K-fold training pipeline
│   ├── evaluate.py             # Metrics and McNemar's test
│   ├── compare_models.py       # Model comparison script
│   ├── hyperparameter_search.py # Grid/random search
│   ├── workflow.py             # End-to-end orchestration
│   ├── data/
│   │   ├── splits.py           # Hierarchical spatial grouping
│   │   ├── dataset.py          # Manifest-based PyTorch Dataset
│   │   ├── augmentation.py     # Training/eval transforms
│   │   ├── quality.py          # Data quality audit
│   │   ├── versioning.py       # Manifest checksumming
│   │   └── download_*.py       # Data source downloaders
│   ├── models/
│   │   ├── baseline_cnn.py     # From-scratch CNN
│   │   └── transfer_models.py  # ResNet50 / EfficientNetV2-S
│   └── explain/
│       ├── gradcam.py          # Grad-CAM implementation
│       └── export_gradcam.py   # Batch heatmap export
├── tests/                      # 7 test modules
├── manifests/                  # Data manifest CSVs
├── config.py                   # Centralised configuration
├── pyproject.toml              # Package definition
└── requirements.txt            # Dependencies
```

## Appendix B: Configuration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| IMAGE_SIZE | 224 | ImageNet-compatible input size |
| BATCH_SIZE | 32 | Training batch size |
| LEARNING_RATE | 1e-4 | Fine-tuning learning rate |
| EPOCHS_DEFAULT | 15 | Training epochs |
| SEED | 42 | Reproducibility |
| N_FOLDS | 5 | Cross-validation folds |
| GRID_CELL_SIZE_DEG | 0.2 | ~20 km spatial grouping |
| MIN_GRID_SIZE | 15 | Fallback threshold for grid cells |
| MIN_DISTRICT_SIZE | 30 | Fallback threshold for districts |
| Confidence threshold | 0.7 | Below this, system abstains |
| Critical threshold | 0.9 | Above this, immediate removal |

## Appendix C: Risk Level Decision Table

```
Input: Prediction(species, confidence)
    │
    ├── confidence < 0.70?
    │       YES → MEDIUM risk, Action: Field verification
    │       NO ──┐
    │            ├── species invasive (per INVASORAS)?
    │            │       NO → NATIVE risk, Action: No action
    │            │       YES ──┐
    │            │             ├── confidence ≥ 0.90?
    │            │             │       YES → CRITICAL risk, Action: Immediate removal
    │            │             │       NO  → HIGH risk, Action: Containment (30 days)
```
