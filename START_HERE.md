# Quick Start

Get InvasiveLens running in under 5 minutes.

---

## 1. Install

```bash
cd path/to/invasivelens
pip install -e ".[dev]"
```

Wait for the installation to finish.

---

## 2. Run

```bash
python -m src.cli survey --image data_raw/floralens/Phragmites_australis/Phragmites_australis_1.jpg
```

You should see output like:

```
MEDIUM RISK: Arundo donax (confidence: 27.3%)
Action: Monitoring
Reasoning: Low confidence detection (27.3%). Requires field verification.
```

The model identified a plant, assessed the confidence, and recommended an action.

---

## 3. What Just Happened

1. The CLI loaded a trained ResNet50 model from `checkpoints/`
2. It ran the input image through the classifier
3. The decision engine checked whether the predicted species is invasive
4. Confidence was below 70%, so it recommended manual verification instead of a management action

---

## Next Steps

- [SIMPLE_START.md](SIMPLE_START.md) — Quick reference for all commands and project structure
- [README.md](README.md) — Full project overview
- [PRESENTATION_TEMPLATE.md](PRESENTATION_TEMPLATE.md) — Slide outlines for the thesis defense
- [DEFENSE_QA.md](DEFENSE_QA.md) — Anticipated defense questions and answers
- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) — Detailed technical write-up
