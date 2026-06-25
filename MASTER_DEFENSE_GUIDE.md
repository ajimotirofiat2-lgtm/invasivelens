# Defense Guide: InvasiveLens

Study notes and key talking points for the thesis defense. This complements the detailed Q&A in [DEFENSE_QA.md](DEFENSE_QA.md) and the full technical write-up in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

---

## The Pitch

"I built InvasiveLens, an AI system that identifies invasive plants from photos and tells environmental teams what to do about them. It's trained on 1,426 real Portuguese plant photos and 1.3 million biodiversity records, achieves 87% accuracy with ResNet50, and goes beyond classification — it assigns risk levels and recommends management actions based on prediction confidence. The system is deployed as a CLI, REST API, and Python library."

Keep it under 30 seconds. Hit the three differentiators: real data, decision-making (not just classification), and production deployment.

---

## Project at a Glance

**What it does:** Photo in → species identification → invasiveness check → risk level → management recommendation out.

**Core numbers:**
- 1,426 training photos (FloraLens), 1.3M occurrence records (GBIF), INVASORAS for species status
- 87% accuracy (ResNet50), 88% (EfficientNetV2-S), 82% (baseline CNN)
- 5-fold spatial cross-validation with hierarchical grouping
- 70% confidence threshold, 90% for CRITICAL risk
- 3 deployment modes: CLI, REST API, library

**Species pairs:** Arundo donax / Phragmites australis, Cortaderia selloana / Ammophila arenaria, Acacia dealbata / Acacia longifolia. Note: both Acacias are invasive in Portugal — the pair is about visual confusion, not invasive-vs-native.

---

## Architecture Overview

```
CLI / API / Library
    ↓
InvasiveLensPredictor (inference.py)
    ↓  loads checkpoint, softmax → confidence + predicted class
Decision Engine (decision_engine.py)
    ↓  invasive lookup from config.py → risk level + action
Output to user
```

The key modules and what to say about each:

- **config.py** — All settings in one place. Species pairs, hyperparameters, spatial stratification params. "Single source of truth."
- **src/train.py** — K-fold training loop. Creates StratifiedGroupKFold splits, trains per fold, saves checkpoints and fold-0 predictions.
- **src/inference.py** — Loads a checkpoint, runs forward pass, returns Prediction with class/confidence/abstain flag/top-3.
- **src/decision_engine.py** — Takes prediction, checks invasiveness, applies confidence thresholds, outputs DecisionResult with risk level and actions.
- **src/data/splits.py** — The spatial cross-validation logic. Hierarchical grouping, StratifiedGroupKFold wrapper.
- **src/evaluate.py** — Metrics (accuracy, macro-F1, per-class) and McNemar's test implementation.

---

## Topics to Know Cold

### Transfer Learning

Started from ImageNet-pretrained ResNet50/EfficientNetV2-S. Replaced the final classification layer (2048→6 for ResNet, 1280→6 for EfficientNet). Fine-tuned all layers at 1e-4 learning rate (10x lower than from-scratch). The low rate prevents destroying pretrained features while allowing adaptation.

Why not freeze early layers? Our domain is different enough from ImageNet that adaptation helps. The empirical evidence is definitive: 82% from scratch vs 87–88% with transfer learning.

### Spatial Cross-Validation

The district-level grouping attempt failed because Lisboa had ~60% of records. Can't split one group across folds, so one fold ended up being 60%+ of the data.

The fix: hierarchical grouping. Grid cells (~20km) → district → macro-region → "Other". Dense areas span many cells and distribute properly across folds. Sparse areas get coarser but still spatial groups.

`StratifiedGroupKFold` — stratified keeps class balance, grouped prevents spatial leakage.

Why it matters: without this, the model could learn location-specific features (lighting, soil colour, camera style) instead of plant morphology. Spatial CV tests real generalisation to unseen regions.

### McNemar's Test

Compares two models on the same test set. Contingency table of agree/disagree patterns. Tests whether the discordant pairs are symmetric or skewed.

Used exact binomial (not chi-square) because small sample size.

Result: ResNet50 vs EfficientNetV2-S → p > 0.05, no significant difference. The 1% accuracy gap is noise.

Both models evaluated on fold 0 (the fixed comparison set). The `compare_models.py` script validates that both used the same val_idx before running the test.

### Decision Engine

The core logic in order:
1. Confidence < 70%? → MEDIUM risk, recommend manual verification (abstain)
2. Species not invasive? → NATIVE, no action
3. Confidence ≥ 90%? → CRITICAL, immediate removal
4. Confidence 70–90%? → HIGH, containment within 30 days

Each risk level maps to specific follow-up actions (document extent, notify authorities, plan removal, etc.).

Species invasiveness comes from INVASORAS (government database), defined in `config.py` as `CANDIDATE_PAIRS`.

### Uncertainty Quantification

Softmax output → probability distribution. Confidence = max probability. Below 70%, system abstains.

The point: confidently wrong predictions are dangerous in environmental management. Better to say "I'm not sure" than to trigger unnecessary plant removal or miss an invasive.

Calibration: a well-calibrated model's 90% confident predictions should be correct ~90% of the time. The inference module tracks calibration data from fold-0 predictions.

---

## Potential Hard Questions

These are the ones that might catch you off guard. See [DEFENSE_QA.md](DEFENSE_QA.md) for the full Q&A list.

**"Is 87% accuracy good enough for production use?"**

For this task, yes — especially because the system doesn't pretend to be certain when it isn't. The confidence threshold means uncertain predictions get flagged for manual review rather than acted on blindly. The 87% is across all 6 species including visually similar pairs, evaluated under spatial CV which is more conservative than random splitting.

**"Why didn't you optimise the confidence threshold?"**

Fair criticism. The 70% threshold is heuristic. Proper optimisation would require a cost-benefit analysis: what's the real cost of removing a native plant (false positive) vs. missing an invasive one (false negative)? That data wasn't available for this project, but it's in the future work section. The 70% value was chosen as a reasonable middle ground.

**"How do you handle the Acacia pair if both species are invasive?"**

The CANDIDATE_PAIRS structure uses "invasive"/"native" keys, but `A. longifolia` is also invasive in Portugal — it's more established/naturalised. The pair exists because field teams confuse the two, and distinguishing them matters for management (different control strategies). The code treats `A. longifolia` as the "native" counterpart for classification purposes, which is a simplification. The config notes document this.

**"What happens if someone uses this on a species you didn't train on?"**

The model will still output a prediction — it'll pick whichever of its 6 classes is closest. The confidence should be low, triggering abstention. But this is an open-set problem the system doesn't explicitly handle. A proper solution would need novelty detection or an "unknown" class, which is future work.

**"Why not use a larger dataset?"**

1,426 images is what was available in FloraLens for these species at the quality level I needed (botanist-verified labels). GBIF has more records but they're occurrence data, not images. I could have scraped additional images from the web, but that introduces label noise. Transfer learning compensates well at this scale — the 87% accuracy shows the approach works.

---

## What to Emphasise

**To committee members focused on methodology:** Spatial cross-validation. The failed district-level attempt and the solution. McNemar's test for statistical rigour.

**To committee members focused on applications:** The decision engine. Uncertainty quantification. The three deployment modes.

**To committee members focused on novelty:** Going beyond classification to decision support. Confidence-based abstention. The combination of spatial CV + transfer learning + decision engine in one end-to-end system.

---

## Demo Command

Have this ready to run:

```bash
python -m src.cli survey --image data_raw/floralens/Phragmites_australis/Phragmites_australis_1.jpg
```

Run it once before the defense to make sure it works. If the output shows a low-confidence prediction with a MEDIUM risk recommendation, that's actually a good demo — it shows the uncertainty quantification working.

---

## Key Files to Know

| File | What to say about it |
|------|---------------------|
| `config.py` | "All configuration centralised here — species pairs, hyperparameters, spatial stratification settings" |
| `src/train.py` | "K-fold CV training loop with geographic splits" |
| `src/inference.py` | "Loads checkpoints, runs inference, returns predictions with confidence" |
| `src/decision_engine.py` | "Converts predictions to risk assessments with actionable recommendations" |
| `src/data/splits.py` | "Hierarchical spatial grouping — this is where the geographic CV logic lives" |
| `src/evaluate.py` | "Metrics computation and McNemar's test" |

For deeper technical details, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md). For the full Q&A prep, see [DEFENSE_QA.md](DEFENSE_QA.md).
