"""
Tests for src/evaluate.py. Includes a sanity check of McNemar's test
against a textbook contingency table with a known, hand-checkable result.

Run with:  pytest tests/test_evaluate.py -v
"""
from src.evaluate import average_metrics_over_folds, compute_metrics, mcnemar_test


def test_compute_metrics_perfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    metrics = compute_metrics(y_true, y_pred, class_names=["a", "b", "c"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    print("Perfect-prediction metrics check passed.")


def test_compute_metrics_known_confusion():
    # 4 samples of class 0, 4 of class 1; model gets all class-0 right but
    # half of class-1 wrong -> easy to hand-verify precision/recall.
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 1, 1, 0, 0]
    metrics = compute_metrics(y_true, y_pred, class_names=["native", "invasive"])
    assert metrics["per_class"]["invasive"]["recall"] == 0.5
    # 6 samples predicted "native" (positions 0-3 correctly, 6-7 incorrectly) -> precision = 4/6
    assert abs(metrics["per_class"]["native"]["precision"] - 4 / 6) < 1e-9
    assert metrics["per_class"]["native"]["recall"] == 1.0
    print("Known-confusion per-class metrics check passed:", metrics["per_class"])


def test_mcnemar_textbook_example():
    """
    Classic worked example (used in many stats textbooks/tutorials):
    contingency table
                        B correct   B wrong
        A correct           59         6
        A wrong              7         8
    Discordant pairs: 6 and 7 -> with continuity correction this should
    give a non-significant p-value (the two models barely disagree).
    We reconstruct y_true/y_pred_a/y_pred_b that produce exactly this table.
    """
    n_both_correct, n_only_a, n_only_b, n_both_wrong = 59, 6, 7, 8
    y_true, pred_a, pred_b = [], [], []

    for _ in range(n_both_correct):
        y_true.append(0); pred_a.append(0); pred_b.append(0)
    for _ in range(n_only_a):
        y_true.append(0); pred_a.append(0); pred_b.append(1)
    for _ in range(n_only_b):
        y_true.append(0); pred_a.append(1); pred_b.append(0)
    for _ in range(n_both_wrong):
        y_true.append(0); pred_a.append(1); pred_b.append(1)

    result = mcnemar_test(y_true, pred_a, pred_b, exact=False)
    assert result["contingency_table"] == [[59, 6], [7, 8]]
    # Hand-computed chi-square with continuity correction:
    # (|6-7|-1)^2 / (6+7) = 0^2/13 = 0.0  ->  p should be ~1.0 (no signal)
    assert result["p_value"] > 0.9, f"Expected p≈1.0 for near-identical models, got {result['p_value']}"
    print("McNemar textbook example check passed:", result)


def test_mcnemar_detects_real_difference():
    """One model clearly better: 0 cases where only A is right, 40 cases
    where only B is right -> should be a highly significant result."""
    y_true = [0] * 100
    pred_a = [1] * 40 + [0] * 60   # A wrong on 40 it shouldn't be
    pred_b = [0] * 100             # B always correct
    result = mcnemar_test(y_true, pred_a, pred_b, exact=True)
    assert result["p_value"] < 0.001
    print("McNemar correctly detects a clear, large difference:", result)


def test_average_metrics_over_folds():
    fold_metrics = [
        {"accuracy": 0.8, "macro_f1": 0.75},
        {"accuracy": 0.82, "macro_f1": 0.77},
        {"accuracy": 0.78, "macro_f1": 0.73},
    ]
    avg = average_metrics_over_folds(fold_metrics)
    assert abs(avg["accuracy_mean"] - 0.8) < 1e-9
    assert avg["n_folds"] == 3
    print("Fold averaging check passed:", avg)


if __name__ == "__main__":
    test_compute_metrics_perfect_predictions()
    test_compute_metrics_known_confusion()
    test_mcnemar_textbook_example()
    test_mcnemar_detects_real_difference()
    test_average_metrics_over_folds()
    print("\nAll evaluate.py tests passed.")
