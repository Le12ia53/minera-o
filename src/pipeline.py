from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detectors import Trace, score_all_methods, top_fraction_flags
from .repair import DataDrivenRepairer, RepairResult


@dataclass(frozen=True)
class ModelFreeOutput:
    """Saída da estratégia que recebe somente traces observados."""

    scores: dict[str, np.ndarray]
    flags: dict[str, np.ndarray]
    repairs: dict[str, list[RepairResult]]
    prototype_counts: dict[str, int]


def run_model_free_pipeline(
    traces: list[Trace],
    alert_rate: float,
    random_state: int = 42,
    n_neighbors: int = 20,
    max_repair_candidates: int | None = None,
    min_prototype_frequency: int = 2,
    max_repair_edit_cost: int | None = 1,
) -> ModelFreeOutput:
    """Detecta e sugere reparos sem receber modelo, rótulos ou cópias limpas.

    Cada método ranqueia o log observado e encaminha a fração definida pelo
    orçamento de inspeção. O reparador aprende seus protótipos exclusivamente
    nos traces que o próprio método não sinalizou.
    """

    scores = score_all_methods(
        traces,
        random_state=random_state,
        n_neighbors=n_neighbors,
    )
    flags: dict[str, np.ndarray] = {}
    repairs: dict[str, list[RepairResult]] = {}
    prototype_counts: dict[str, int] = {}
    for method, method_scores in scores.items():
        method_flags = top_fraction_flags(method_scores, alert_rate)
        reference = [trace for trace, flag in zip(traces, method_flags) if not flag]
        repairer = DataDrivenRepairer(
            max_candidates=max_repair_candidates,
            min_frequency=min_prototype_frequency,
            max_edit_cost=max_repair_edit_cost,
        ).fit(reference)
        flags[method] = method_flags
        repairs[method] = repairer.repair_if_flagged(traces, method_flags.tolist())
        prototype_counts[method] = repairer.prototype_count
    return ModelFreeOutput(scores, flags, repairs, prototype_counts)
