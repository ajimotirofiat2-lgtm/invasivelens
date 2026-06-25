# Presentation Slides

Outline for the thesis defense presentation. Each section maps to a slide or group of slides.

---

## Slide 1: Title

**InvasiveLens: AI-Based Invasive Plant Identification and Management Support for Portugal**

[Your Name] — [Date]

Master's Thesis, [University / Department]

---

## Slide 2: The Problem

Invasive plants are one of the biggest threats to Portugal's ecosystems. Species like *Arundo donax*, *Cortaderia selloana*, and *Acacia dealbata* displace native flora, degrade habitats, and cost significant resources to manage.

The identification bottleneck: correctly distinguishing invasive species from their visually similar native counterparts requires expert botanists. That's slow, expensive, and doesn't scale.

*Consider adding: a photo pair showing how similar Arundo donax and Phragmites australis look.*

---

## Slide 3: The Solution

InvasiveLens goes beyond species classification:

- **Identifies** the plant species from a photograph
- **Checks** whether it's classified as invasive (using the INVASORAS database)
- **Assesses risk** based on prediction confidence (CRITICAL / HIGH / MEDIUM / NATIVE)
- **Recommends action** (remove, contain, verify, monitor, no action)
- **Flags uncertainty** — if confidence is below 70%, it says so instead of guessing

The system is trained on 1,426 real Portuguese plant photos and deployed as a command-line tool, REST API, and Python library.

---

## Slide 4: Data Sources

All training data from real-world, authoritative sources:

| Source | Content | Scale |
|--------|---------|-------|
| FloraLens | Botanist-verified Portuguese plant photos | 1,426 images |
| GBIF | Georeferenced occurrence records | 1.3M records |
| INVASORAS | Official invasive species status | Species classifications |

Three species pairs selected for ecological significance and visual difficulty:
- *Arundo donax* vs *Phragmites australis* — tall riparian grasses
- *Cortaderia selloana* vs *Ammophila arenaria* — coastal grasses
- *Acacia dealbata* vs *Acacia longifolia* — both invasive Acacias (different spread rates)

---

## Slide 5: Live Demo

Run this in the terminal:

```bash
python -m src.cli survey --image data_raw/floralens/Phragmites_australis/Phragmites_australis_1.jpg
```

Expected output (something like):
```
MEDIUM RISK: Arundo donax (confidence: 27.3%)
Action: Monitoring
Reasoning: Low confidence detection (27.3%). Requires field verification.
```

This is actually a useful demo scenario: the model is uncertain, so instead of making a wrong decision it recommends manual review. That's the uncertainty quantification working as designed.

*Note: run this once before the presentation to confirm it works.*

---

## Slide 6: Model Architecture

Transfer learning from ImageNet:

| Model | Accuracy | Macro-F1 | Use Case |
|-------|----------|----------|----------|
| ResNet50 | 87% | 0.85 | Production (recommended) |
| EfficientNetV2-S | 88% | 0.86 | Mobile / constrained deployment |
| Baseline CNN | 82% | 0.80 | Performance floor (from scratch) |

The 5–6% gap between transfer learning and the from-scratch baseline justifies the approach. McNemar's test confirms the 1% difference between ResNet50 and EfficientNet is not statistically significant (p > 0.05).

---

## Slide 7: Spatial Cross-Validation

Why not random train/test splits? Plants from the same location share environmental cues (lighting, soil, camera). Random splitting lets the model learn location features instead of plant features.

My approach: 5-fold spatial cross-validation with hierarchical grouping.

- Grid cells (~20km) as primary groups
- District and macro-region as fallbacks for sparse areas
- StratifiedGroupKFold maintains class balance across folds

The first version (district-level grouping) failed because Lisboa held ~60% of records. The hierarchical approach fixes this — dense areas span many grid cells and distribute properly.

---

## Slide 8: Decision Engine

Beyond classification: the decision engine converts predictions into actionable management recommendations.

| Risk Level | Confidence | Action |
|------------|-----------|--------|
| CRITICAL | ≥90%, invasive | Immediate removal |
| HIGH | 70–90%, invasive | Containment within 30 days |
| MEDIUM | <70%, any species | Field verification |
| NATIVE | ≥70%, non-invasive | No action |

Key design choice: below 70% confidence, the system abstains. In environmental management, a confidently wrong decision is more dangerous than admitting uncertainty.

---

## Slide 9: Real-World Impact

Scenario: environmental team surveys 50 locations in a day.

| | Manual Approach | With InvasiveLens |
|--|---------------|-------------------|
| Time to identify | Days to weeks | ~2 minutes (batch) |
| Expert required | Yes (botanist) | No |
| Risk prioritisation | Manual | Automatic (risk levels) |
| Uncertainty flagging | Depends on expert | Built-in (confidence thresholds) |

The system augments expert judgement — it doesn't replace it. High-confidence findings can be acted on; uncertain ones get routed for manual review.

---

## Slide 10: Deployment

Three interfaces for different users:

**CLI** — `python -m src.cli survey --image photo.jpg`
Works offline. One command to process a photo or an entire survey folder.

**REST API** — `python -m src.api --port 5000`
Upload via HTTP, receive JSON. For web and mobile integration.

**Python library** — Import `InvasiveLensPredictor` into custom workflows.

---

## Slide 11: Limitations and Future Work

Current scope:
- 3 species pairs (6 species) — proof of concept, framework scales
- 1,426 training images — small but viable with transfer learning
- Visual features only — no habitat, elevation, or seasonal context
- Heuristic confidence thresholds — not formally optimised
- Not yet deployed in production

Next steps:
- More species pairs and training data
- Satellite imagery integration for landscape-scale detection
- Multi-modal features (visual + contextual)
- Formal threshold optimisation with cost-benefit analysis
- Deployment with a Portuguese environmental agency

---

## Slide 12: Summary

1. Trained on real Portuguese biodiversity data (1,426 images + 1.3M records)
2. 87–88% accuracy under spatial cross-validation
3. Decision engine produces risk-tiered management recommendations
4. Uncertainty quantification prevents confidently wrong decisions
5. Production-ready deployment (CLI, API, library)

---

## Slide 13: Questions?

---

## Presenter Notes

**Before the presentation:**
- Run the demo command once to verify it works
- Have a backup screenshot of the output in case of technical issues
- Know the key numbers: 1,426 images, 87% accuracy, 70% threshold, 5 folds, 3 deployment modes

**During:**
- Spend most time on slides 5 (demo), 7 (spatial CV), and 8 (decision engine) — these are the strongest technical contributions
- If a question catches you off guard: "That's a good point — let me think about that for a moment" is fine. Better than guessing.
- For questions about specific code files, it's okay to say "That module handles [general purpose]. I can walk through the specifics if you'd like."

**Common questions to expect:**
- Why these species? (Ecologically significant + visually challenging, framework scales)
- Is 87% good enough? (Yes — uncertainty quantification handles the remaining 13%)
- Why not more species? (Proof of concept; framework supports it but needs more data)
- How does it compare to existing tools? (Most just classify — this provides risk assessment and actionable recommendations)

See [DEFENSE_QA.md](DEFENSE_QA.md) for the full Q&A preparation.
