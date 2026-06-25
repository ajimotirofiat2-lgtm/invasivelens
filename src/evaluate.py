"""
Evaluation metrics for InvasiveLens.

McNemar's test specifically requires the two models' predictions to be on
the EXACT same paired samples (see splits.py — use result['mcnemar_split']
test indices for both models, nothing else). It is not meaningful applied
across pooled/aggregated cross-validation folds.
"""
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from statsmodels.stats.contingency_tables import mcnemar


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> dict:
    """Primary metrics: overall accuracy, macro-F1, and per-class P/R/F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0,
    )
    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
    }


def average_metrics_over_folds(fold_metrics: list[dict]) -> dict:
    """Average accuracy/macro_f1 across CV folds, weighted by fold size if
    `support` totals are available; simple mean otherwise."""
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "n_folds": len(fold_metrics),
    }


def mcnemar_test(
    y_true: Sequence[int],
    y_pred_model_a: Sequence[int],
    y_pred_model_b: Sequence[int],
    exact: bool = True,
) -> dict:
    """
    McNemar's test on a SINGLE shared test set (e.g. result['mcnemar_split']
    test indices from splits.py), comparing whether model A and model B
    disagree with the ground truth in a way that's symmetric (no real
    difference) or skewed (one model is genuinely better).

    Returns the 2x2 contingency table plus the test statistic and p-value.
    Uses the exact binomial test by default (recommended when the
    discordant-pair count is small, which is plausible at our dataset
    scale); set exact=False for the chi-square approximation with
    continuity correction on larger discordant counts.
    """
    y_true = np.asarray(y_true)
    a_correct = np.asarray(y_pred_model_a) == y_true
    b_correct = np.asarray(y_pred_model_b) == y_true

    both_correct = int(np.sum(a_correct & b_correct))
    only_a_correct = int(np.sum(a_correct & ~b_correct))
    only_b_correct = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))

    table = [[both_correct, only_a_correct], [only_b_correct, both_wrong]]
    result = mcnemar(table, exact=exact, correction=True)

    return {
        "contingency_table": table,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_discordant_pairs": only_a_correct + only_b_correct,
        "interpretation": (
            "Models differ significantly (p < 0.05)."
            if result.pvalue < 0.05
            else "No statistically significant difference detected (p >= 0.05)."
        ),
    }
