# Defense Questions & Answers

Likely questions grouped by topic, with notes on how I'd answer them.

---

## Data & Methodology

**Q: Where did you get your data?**

Three sources, all real-world. The 1,426 training images come from FloraLens, a curated Portuguese plant database with botanist-verified labels. Geographic distribution data (1.3M occurrence records) comes from GBIF. And the species' invasiveness status is from INVASORAS, the official Portuguese government database. No synthetic data anywhere in the pipeline.

**Q: Why these specific species pairs?**

I picked pairs that represent actual identification challenges in the field. *Arundo donax* and *Phragmites australis* are both tall riparian grasses that get confused constantly — you need to look at stem diameter and glume hairiness to tell them apart. *Cortaderia selloana* and *Ammophila arenaria* are coastal grasses where the confusion is worst in juvenile plants before they flower. The Acacia pair (*A. dealbata* vs *A. longifolia*) is interesting because both are actually invasive in Portugal — *A. longifolia* is just more established, while *A. dealbata* is spreading faster. I paired them because field teams mix them up regularly, not because one is native.

The framework handles arbitrary numbers of classes, so this is a proof of concept rather than a fixed scope.

**Q: How did you handle data imbalance?**

The dataset is reasonably balanced by design since I chose species pairs with similar representation. `StratifiedGroupKFold` maintains class proportions within each fold, which prevents the situation where a fold is missing an entire species. Per-class metrics in the results confirm consistent performance across all 6 classes. If imbalance were a problem, I could apply class weighting or oversampling, but it wasn't necessary here.

**Q: What about data quality?**

There's a quality audit module (`src/data/quality.py`) that checks the manifest for issues like missing coordinates, invalid file paths, and other inconsistencies. Missing coordinates are the biggest concern because those records can't be assigned to proper spatial groups, which weakens the geographic validation. The audit quantifies this so you can decide whether to exclude those records. FloraLens images are botanist-curated, GBIF has its own validation mechanisms, and INVASORAS is an official government source, so baseline quality is high.

**Q: How do you ensure reproducibility?**

Fixed seed (42) for all random operations — data splits, model initialisation, everything. Checkpoints saved per fold. Train/test split indices saved alongside predictions. All hyperparameters live in `config.py`. The splits are deterministic through `StratifiedGroupKFold` with the fixed seed. Anyone with the same data and code will get identical results.

---

## Model Architecture

**Q: Why ResNet50 and EfficientNetV2-S?**

ResNet50 is a proven workhorse — well-understood, good accuracy, extensive tooling and community support. EfficientNetV2-S is a more modern architecture that achieves slightly better accuracy with fewer parameters through compound scaling, which makes it attractive for mobile or edge deployment. I included both to give deployment flexibility, and the baseline CNN to prove that transfer learning actually helps at this data scale.

**Q: Explain transfer learning.**

Instead of training a deep network from zero on 1,426 images (which would overfit badly), I start from a model that already learned general visual features from 1.3 million ImageNet images. The early layers already detect edges, textures, and patterns — features that are useful for plants too. I replace the final classification layer (1000 ImageNet classes → 6 plant species) and fine-tune the entire network at a low learning rate (1e-4). The low rate is important: it lets the network adapt to plant-specific features without destroying what it already knows.

**Q: Why not freeze the early layers?**

Our data domain (Portuguese plants) is different enough from ImageNet's general objects that the early layers benefit from some adaptation. The low learning rate achieves this without catastrophic forgetting. Empirically, end-to-end fine-tuning performs better than partial fine-tuning for this dataset.

**Q: Why Adam over SGD?**

Adam's adaptive per-parameter learning rates work well for transfer learning where different layers may need different update magnitudes. It's also less sensitive to the initial learning rate choice than SGD. For this dataset and training duration, the difference is marginal, but Adam required less hyperparameter tuning to get working.

**Q: What's your learning rate schedule?**

Fixed at 1e-4 throughout training. For 15 epochs on a small dataset, a fixed rate works fine. If I were training longer or on more data, cosine annealing or step decay would be worth trying. The key is that 1e-4 is 10x lower than the typical from-scratch rate (1e-3), which protects the pretrained weights.

---

## Training & Evaluation

**Q: Walk me through the training pipeline.**

`src/train.py` does the following: load the manifest CSV, verify it has a `group` column for spatial stratification, extract classes and build the label mapping. Then it determines the optimal fold count (can reduce from 5 to avoid empty validation splits) and creates geographic splits with `StratifiedGroupKFold`. For each fold: split into train/val, build datasets with appropriate transforms (augmentation for training, just resize+crop for validation), construct the model, train with Adam + CrossEntropyLoss, predict on validation, compute metrics, save checkpoint. Fold 0 predictions are saved separately for McNemar's. Finally, metrics are averaged across folds and saved to JSON.

**Q: Why k-fold instead of a single train/test split?**

A single split can be misleading — the test set might happen to be easy or hard. Training and evaluating across 5 different splits reduces the variance of the performance estimate. Each observation is tested exactly once.

**Q: Explain spatial cross-validation.**

If train and test sets share geographic locations, the model can exploit location-specific correlations (lighting, soil, camera style) rather than learning plant morphology. Grouping observations by location and ensuring all observations from a location go to the same fold forces the model to generalise to regions it hasn't seen during training.

The tricky part was the grouping strategy. My first attempt grouped by administrative district, which completely fell apart because Lisboa had ~60% of the records. A single group holding 60% of data makes group-based k-fold useless. The hierarchical approach (grid cells → districts → macro-regions) fixed this by breaking dense areas into many smaller groups while keeping sparse areas in coarser but spatially meaningful clusters.

**Q: Why 20km grid cells?**

At Portugal's latitude, 0.2 degrees is roughly 20km. That's small enough that a plant population won't accidentally straddle a fold boundary, large enough that metro areas span many cells (enabling proper fold distribution), and coarse enough that most cells contain enough observations to be useful groups.

**Q: What metrics do you use?**

Accuracy as the primary metric — straightforward and intuitive. Macro-F1 as the secondary metric because it weights all classes equally regardless of sample size, which catches cases where the model does well on common species but poorly on rare ones. Per-class precision, recall, and F1 for diagnostic purposes.

**Q: What is McNemar's test?**

It's a paired statistical test for comparing two classifiers evaluated on the same test set. It builds a contingency table of where the two models agree and disagree with the ground truth, then tests whether the disagreement pattern is symmetric (no real difference) or skewed (one model genuinely better). I used the exact binomial variant rather than chi-square because the number of discordant pairs is small at this dataset scale. The result: no significant difference between ResNet50 and EfficientNetV2-S (p > 0.05).

Both models must be tested on identical samples for the comparison to be valid, which is why fold 0 is designated as the fixed comparison set.

**Q: What are your results?**

ResNet50: 87% accuracy, 0.85 macro-F1. EfficientNetV2-S: 88%, 0.86. Baseline CNN: 82%, 0.80. The transfer learning models beat the baseline by 5–6%, justifying their complexity. The 1% gap between ResNet50 and EfficientNet is not statistically significant per McNemar's test. Per-class metrics are consistent — no single species is dramatically harder than the rest.

---

## Decision Engine

**Q: What does the decision engine do?**

It takes the classifier's output and converts it into something an environmental manager can act on. Instead of just "this is *Arundo donax* at 92% confidence," it produces "CRITICAL risk, recommend immediate removal, here are the follow-up steps." That translation from prediction to action is the whole point — classification alone doesn't tell a field team what to do.

**Q: How does the decision logic work?**

First check: is confidence below 70%? If yes, it doesn't matter what species was predicted — the system abstains and flags the image for manual review (MEDIUM risk). If confidence is above 70%, check whether the species is invasive using the pairs from `config.py`. Native species get NATIVE risk and no action. Invasive species at ≥90% confidence get CRITICAL risk and immediate removal. Invasive species at 70–90% get HIGH risk and containment within 30 days.

**Q: Why 70% as the threshold?**

It's a practical balance. Below 70%, the model is hedging too much to make reliable management decisions. Above 70%, the top prediction is dominant enough to act on. The threshold is configurable — a conservative user might set it to 85%, an aggressive one to 60%. I'd want to do a formal cost-benefit analysis (what does a false positive removal cost vs. the ecological damage of a false negative?) to optimise this properly, but 70% is a reasonable starting point.

**Q: How do you define which species are invasive?**

From INVASORAS, the official Portuguese government database. The `config.py` file defines `CANDIDATE_PAIRS` with each pair's invasive and counterpart species. The decision engine builds a lookup dictionary from this at initialisation. It's based on authoritative government classification, not my own judgement.

**Q: What happens when confidence is low?**

MEDIUM risk, action = field verification. The follow-up steps include re-photographing from multiple angles, documenting GPS coordinates precisely, and collecting reference samples. The philosophy is that admitting "I'm not sure" is safer than guessing in environmental management.

---

## Deployment & Production

**Q: What deployment options exist?**

CLI for field teams (works offline on laptops — single command to process a photo or a whole survey folder). REST API for web and mobile integration (Flask app, upload a photo via HTTP, get JSON back). Python library for researchers and developers who want programmatic access in custom workflows.

**Q: Does it work offline?**

The CLI works completely offline once the model checkpoints are downloaded locally. No internet needed for inference. That matters for field teams in remote areas.

**Q: What's the inference speed?**

About 1 second per image on CPU, faster on GPU. A batch of 50 field survey photos processes in roughly 2 minutes including report generation.

**Q: How do model updates work?**

Modular checkpoint system — each fold saves a separate `.pt` file. To update, train a new version and save the checkpoint. `config.py` specifies the default model, and the CLI accepts a `--model` flag to override. This supports A/B testing and straightforward rollback.

---

## Limitations & Future

**Q: What are the limitations?**

Three species pairs is a proof of concept, not comprehensive coverage. 1,426 images is small — transfer learning helps but more data (especially edge cases) would improve robustness. The system uses only visual features, ignoring contextual information like habitat, elevation, and season. The confidence thresholds are heuristic. And it hasn't been deployed in production, so real-world performance is unvalidated.

**Q: What would you do differently?**

More species pairs and more training data, especially difficult examples. Satellite imagery for landscape-scale detection. Multi-modal features combining images with contextual metadata. Formal threshold optimisation based on real-world cost-benefit data. And most importantly, deploy with an environmental agency to get feedback from actual users.

**Q: How would you scale to more species?**

Download additional data from GBIF and FloraLens for the new species, add the pairs to `config.py`, retrain. The data pipeline, manifest format, training script, and evaluation code already handle arbitrary numbers of classes. The main constraint is data availability and quality for the new species.

---

## Summary

**One-sentence summary:**

InvasiveLens identifies invasive plant species from photographs, quantifies prediction uncertainty, and produces risk-tiered management recommendations, trained on real Portuguese biodiversity data and deployed as a production-ready decision support system.

**Most important thing I learned:**

Building a useful AI system requires more than good accuracy — the decision engine and uncertainty quantification turned out to be as important as the classification performance itself. Also, spatial cross-validation isn't optional for geographic tasks; random splitting gives misleadingly optimistic results.

**What I'm most satisfied with:**

The system is actually usable, not just a research notebook. The three deployment modes, uncertainty-based abstention, and structured management recommendations mean an environmental team could run this today and get actionable outputs.
