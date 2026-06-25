# InvasiveLens — Quick Reference

A concise overview of what the project does, how to run it, and where things live.

---

## What It Does

InvasiveLens takes a plant photograph as input and outputs:
- The identified species
- A confidence score (0–100%)
- Whether the species is invasive in Portugal
- A risk level (CRITICAL, HIGH, MEDIUM, LOW, NATIVE)
- A recommended management action (remove, contain, monitor, verify, no action)

If the model isn't confident enough (below 70%), it flags the image for manual review instead of making a decision.

---

## Running It

### Single photo
```bash
python -m src.cli survey --image your_photo.jpg
```

### Batch processing (whole folder)
```bash
python -m src.cli survey --batch survey_photos/ --export results.csv
```

### Via web API
```bash
python -m src.api --port 5000
```
Then upload photos through `http://localhost:5000/predict`.

### Quick species ID (classification only, no decision engine)
```bash
python -m src.cli identify your_photo.jpg
```

---

## Example Output

```bash
python -m src.cli survey --image data_raw/floralens/Phragmites_australis/Phragmites_australis_1.jpg
```

```
MEDIUM RISK: Arundo donax (confidence: 27.3%)
Action: Monitoring
Reasoning: Low confidence detection (27.3%). Requires field verification.
```

The model thinks this might be *Arundo donax* but is only 27% confident, so it recommends manual verification rather than triggering a management action.

---

## Training Data

- **1,426 real plant photos** from FloraLens (Portuguese plant database, botanist-verified)
- **1.3 million biodiversity records** from GBIF (geographic distribution data)
- **INVASORAS** — official Portuguese government invasive species database

The model (ResNet50) achieves 87% accuracy on these images under spatial cross-validation.

---

## Project Structure

```
invasivelens/
├── src/                          # All source code
│   ├── cli.py                    # Command-line interface (what you run)
│   ├── api.py                    # REST API for web/mobile
│   ├── inference.py              # Model loading and prediction
│   ├── decision_engine.py        # Risk assessment logic
│   ├── train.py                  # Training pipeline
│   └── data/                     # Data loading, splits, augmentation
│
├── config.py                     # All settings (species pairs, hyperparameters)
├── checkpoints/                  # Saved model weights
│   ├── resnet50_fold0.pt         # ResNet50 (recommended)
│   └── ...
├── data_raw/                     # Raw training images
│   └── floralens/                # 1,426 Portuguese plant photos
├── manifests/                    # CSV manifests linking images to labels
└── results/                      # Metrics and predictions
```

Key files to know:
- `src/cli.py` — the interface you interact with
- `checkpoints/resnet50_fold*.pt` — the trained model weights
- `config.py` — species pairs, thresholds, and all configuration

---

## Installation

```bash
cd path/to/invasivelens
pip install -e ".[dev]"
```

---

## Quick Reference Table

| Task | Command |
|------|---------|
| Assess one photo | `python -m src.cli survey --image photo.jpg` |
| Assess a folder | `python -m src.cli survey --batch folder/` |
| Start web API | `python -m src.api --port 5000` |
| Check model calibration | `python -m src.cli calibration --model resnet50` |
| Train a model | `python -m src.train --manifest manifests/combined_manifest.csv --model resnet50` |

---

## Further Documentation

- [README.md](README.md) — Full project overview
- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) — Detailed technical write-up
- [PRESENTATION_TEMPLATE.md](PRESENTATION_TEMPLATE.md) — Slide outlines for the defense
- [DEFENSE_QA.md](DEFENSE_QA.md) — Anticipated defense questions and answers
- [MASTER_DEFENSE_GUIDE.md](MASTER_DEFENSE_GUIDE.md) — Study notes for the defense
