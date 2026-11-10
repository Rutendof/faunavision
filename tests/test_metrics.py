import numpy as np

from faunavision.metrics import (
    cluster_report,
    expected_calibration_error,
    fit_temperature,
    macro_f1,
    softmax,
    summarize,
    topk_accuracy,
)

NAMES = ["white_rhino", "black_rhino", "lion"]
CLUSTERS = {"white_rhino": "rhinos", "black_rhino": "rhinos", "lion": "large_cats"}


def test_topk():
    probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    y = np.array([1, 2])
    assert topk_accuracy(probs, y, 1) == 0.5
    assert topk_accuracy(probs, y, 2) == 1.0


def test_macro_f1_perfect_and_skips_absent_classes():
    y = np.array([0, 0, 1])
    assert macro_f1(y.copy(), y, n_classes=3) == 1.0


def test_ece_zero_when_confidence_matches_accuracy():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, np.array([0, 1])) == 0.0


def test_temperature_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(size=(2000, 3))
    logits[np.arange(2000), y] += 1.0  # weakly informative...
    logits *= 8  # ...but wildly overconfident
    t = fit_temperature(logits, y)
    assert t > 3
    assert expected_calibration_error(softmax(logits, t), y) < expected_calibration_error(softmax(logits), y)


def test_cluster_report_counts_lookalike_errors():
    y = np.array([0, 0, 0, 2])
    pred = np.array([0, 1, 2, 2])  # one sibling error, one cross-cluster error
    rep = cluster_report(pred, y, NAMES, CLUSTERS)
    assert rep["rhinos"]["accuracy"] == 1 / 3
    assert rep["rhinos"]["lookalike_share_of_errors"] == 0.5
    assert rep["large_cats"]["accuracy"] == 1.0


def test_pair_accuracy_ignores_errors_outside_the_pair():
    from faunavision.metrics import pair_accuracy
    y = np.array([0, 0, 1, 1, 0])
    pred = np.array([0, 1, 1, 1, 2])  # last one went to lion: excluded
    acc, n = pair_accuracy(pred, y, 0, 1)
    assert n == 4 and acc == 0.75


def test_summarize_near_miss_share():
    y = np.array([0, 0, 2])
    logits = np.eye(3)[[1, 2, 2]] * 5  # rhino->rhino (near miss), rhino->lion
    s = summarize(logits, y, NAMES, CLUSTERS)
    assert s["near_miss_share_of_errors"] == 0.5
    assert set(s["per_class"]) == {"white_rhino", "lion"}
