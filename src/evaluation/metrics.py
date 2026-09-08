"""Evaluation metrics for MisconceptDialogue experiments."""

import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def macro_f1(y_true: list, y_pred: list, labels: list | None = None) -> float:
    """Compute macro F1 score (BEA 2025 primary metric)."""
    return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)


def accuracy_at_k(y_true: list, y_pred_ranked: list[list], k: int = 1) -> float:
    """Compute Accuracy@k — fraction where true label is in top-k predictions."""
    correct = 0
    for true, preds in zip(y_true, y_pred_ranked):
        if true in preds[:k]:
            correct += 1
    return correct / len(y_true) if y_true else 0.0


def mean_average_precision_at_k(
    y_true: list, y_pred_ranked: list[list], k: int = 25
) -> float:
    """Compute MAP@k (EEDI Kaggle metric)."""
    aps = []
    for true, preds in zip(y_true, y_pred_ranked):
        preds_k = preds[:k]
        if true not in preds_k:
            aps.append(0.0)
        else:
            rank = preds_k.index(true) + 1
            aps.append(1.0 / rank)
    return np.mean(aps) if aps else 0.0


def mrr(y_true: list, y_pred_ranked: list[list]) -> float:
    """Compute Mean Reciprocal Rank."""
    rrs = []
    for true, preds in zip(y_true, y_pred_ranked):
        if true in preds:
            rank = preds.index(true) + 1
            rrs.append(1.0 / rank)
        else:
            rrs.append(0.0)
    return np.mean(rrs) if rrs else 0.0


def remediation_success_rate(
    pre_correct: list[bool], post_correct: list[bool]
) -> float:
    """Compute Remediation Success Rate.

    RSR = fraction of initially-wrong answers that become correct after remediation.
    """
    wrong_indices = [i for i, c in enumerate(pre_correct) if not c]
    if not wrong_indices:
        return 0.0
    fixed = sum(1 for i in wrong_indices if post_correct[i])
    return fixed / len(wrong_indices)


def answer_reveal_rate(responses: list[str], gold_answers: list[str]) -> float:
    """Compute Answer Reveal Rate — fraction of responses that contain the answer.

    Lower is better for pedagogical quality.
    """
    revealed = 0
    for resp, answer in zip(responses, gold_answers):
        # Normalize for comparison
        answer_normalized = answer.strip().lower()
        resp_normalized = resp.strip().lower()
        if answer_normalized in resp_normalized:
            revealed += 1
    return revealed / len(responses) if responses else 0.0


def actionability_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Compute Actionability F1 (BEA 2025 Track 4 metric).

    Binary: 1 = actionable feedback, 0 = not actionable.
    """
    return f1_score(y_true, y_pred, average="binary", zero_division=0)


def guidance_quality_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Compute Guidance Quality F1 (BEA 2025 Track 3 metric)."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def strategy_appropriateness(
    predicted_strategies: list[str],
    optimal_strategies: list[str],
) -> float:
    """Compute Strategy Appropriateness Score.

    Fraction of times the predicted strategy matches the optimal one.
    """
    correct = sum(
        1 for p, o in zip(predicted_strategies, optimal_strategies) if p == o
    )
    return correct / len(predicted_strategies) if predicted_strategies else 0.0


def compute_all_detection_metrics(y_true: list, y_pred: list) -> dict:
    """Compute all detection metrics."""
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def compute_all_diagnosis_metrics(
    y_true: list, y_pred_ranked: list[list]
) -> dict:
    """Compute all diagnosis metrics."""
    return {
        "acc_at_1": accuracy_at_k(y_true, y_pred_ranked, k=1),
        "acc_at_5": accuracy_at_k(y_true, y_pred_ranked, k=5),
        "map_at_25": mean_average_precision_at_k(y_true, y_pred_ranked, k=25),
        "mrr": mrr(y_true, y_pred_ranked),
    }


def bootstrap_ci(
    metric_fn, y_true, y_pred, n_bootstrap: int = 10000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Returns: (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        scores.append(metric_fn(yt, yp))
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1 - alpha))
    return point, lower, upper


def ar_excluded_rsr(
    pre_correct: list[bool],
    post_correct: list[bool],
    answer_revealed: list[bool],
) -> float:
    """RSR excluding cases where the tutor revealed the answer.

    Only counts a remediation as successful if:
    1. The student was initially wrong (pre_correct=False)
    2. The student is now correct (post_correct=True)
    3. The tutor did NOT reveal the answer (answer_revealed=False)

    Returns:
        Fraction of initially-wrong answers that became correct WITHOUT
        the answer being revealed. Returns 0.0 if no wrong answers exist.
    """
    wrong_indices = [i for i, c in enumerate(pre_correct) if not c]
    if not wrong_indices:
        return 0.0
    genuine = sum(
        1 for i in wrong_indices
        if post_correct[i] and not answer_revealed[i]
    )
    return genuine / len(wrong_indices)


def rsr_with_ci(
    pre_correct: list[bool],
    post_correct: list[bool],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """RSR with bootstrap confidence interval.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    point = remediation_success_rate(pre_correct, post_correct)
    rng = np.random.RandomState(seed)
    n = len(pre_correct)
    if n == 0:
        return (0.0, 0.0, 0.0)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        pre_b = [pre_correct[i] for i in idx]
        post_b = [post_correct[i] for i in idx]
        scores.append(remediation_success_rate(pre_b, post_b))
    alpha = (1 - ci) / 2
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1 - alpha)))
    return (point, lower, upper)
