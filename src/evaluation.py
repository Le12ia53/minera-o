from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

from sklearn.metrics import average_precision_score, roc_auc_score

from .repair import Trace, damerau_levenshtein


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))

    def safe(num: float, den: float) -> float:
        return num / den if den else 0.0

    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    accuracy = safe(tp + tn, tp + tn + fp + fn)
    f1 = safe(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
    }


def ranking_metrics(y_true: list[int], scores: list[float]) -> dict[str, float]:
    """Avalia a ordenação dos casos sem depender de um limiar normativo."""

    if len(y_true) != len(scores):
        raise ValueError("y_true e scores devem ter o mesmo tamanho.")
    if len(set(y_true)) < 2:
        return {"average_precision": 0.0, "roc_auc": 0.0}
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Não há linhas para gravar.")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def repair_metrics(
    original: list[Trace],
    observed: list[Trace],
    repaired: list[Trace],
    anomaly_labels: list[int],
    reference_variants: set[Trace] | None = None,
) -> dict[str, float | int]:
    """Avalia reparo; referência normativa é opcional e usada somente após o teste."""

    if not (len(original) == len(observed) == len(repaired) == len(anomaly_labels)):
        raise ValueError("As listas de avaliação devem ter o mesmo tamanho.")
    anomaly_positions = [index for index, label in enumerate(anomaly_labels) if label]
    normal_positions = [index for index, label in enumerate(anomaly_labels) if not label]
    before = [damerau_levenshtein(observed[i], original[i]) for i in anomaly_positions]
    after = [damerau_levenshtein(repaired[i], original[i]) for i in anomaly_positions]
    recovered = sum(repaired[i] == original[i] for i in anomaly_positions)
    normals_preserved = sum(repaired[i] == original[i] for i in normal_positions)
    improved = sum(after_value < before_value for before_value, after_value in zip(before, after))
    worsened = sum(after_value > before_value for before_value, after_value in zip(before, after))
    result: dict[str, float | int] = {
        "anomalous_cases": len(anomaly_positions),
        "exactly_recovered": recovered,
        "exact_recovery_rate": recovered / len(anomaly_positions) if anomaly_positions else 0.0,
        "improved_cases": improved,
        "improvement_rate": improved / len(anomaly_positions) if anomaly_positions else 0.0,
        "worsened_cases": worsened,
        "worsening_rate": worsened / len(anomaly_positions) if anomaly_positions else 0.0,
        "mean_distance_before": mean(before) if before else 0.0,
        "mean_distance_after": mean(after) if after else 0.0,
        "normal_preservation_rate": (
            normals_preserved / len(normal_positions) if normal_positions else 0.0
        ),
    }
    if reference_variants is not None:
        valid = sum(trace in reference_variants for trace in repaired)
        result["oracle_validity_after_rate"] = valid / len(repaired) if repaired else 0.0
    return result
