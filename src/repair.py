from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


Trace = tuple[str, ...]


def damerau_levenshtein(source: Trace, target: Trace) -> int:
    """Distância com inserção, remoção, substituição e transposição adjacente."""

    rows = len(source) + 1
    cols = len(target) + 1
    matrix = [[0] * cols for _ in range(rows)]
    for row in range(rows):
        matrix[row][0] = row
    for col in range(cols):
        matrix[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            substitution = 0 if source[row - 1] == target[col - 1] else 1
            matrix[row][col] = min(
                matrix[row - 1][col] + 1,
                matrix[row][col - 1] + 1,
                matrix[row - 1][col - 1] + substitution,
            )
            if (
                row > 1
                and col > 1
                and source[row - 1] == target[col - 2]
                and source[row - 2] == target[col - 1]
            ):
                matrix[row][col] = min(matrix[row][col], matrix[row - 2][col - 2] + 1)
    return matrix[-1][-1]


@dataclass(frozen=True)
class RepairResult:
    original_trace: Trace
    repaired_trace: Trace
    edit_cost: int
    prototype_frequency: int
    repair_applied: bool


class DataDrivenRepairer:
    """Sugere reparo usando apenas traces não sinalizados do próprio log.

    Não há consulta a BPMN, Rede de Petri ou lista normativa. Os protótipos são
    variantes observadas com frequência mínima configurável. Entre elas, o
    reparador escolhe a mais próxima por Damerau--Levenshtein e usa frequência
    apenas para desempatar.
    """

    def __init__(
        self,
        max_candidates: int | None = None,
        min_frequency: int = 2,
        max_edit_cost: int | None = 1,
    ):
        if min_frequency < 1:
            raise ValueError("min_frequency deve ser pelo menos 1.")
        self.max_candidates = max_candidates
        self.min_frequency = min_frequency
        self.max_edit_cost = max_edit_cost
        self.frequencies: Counter[Trace] = Counter()
        self.variants: list[Trace] = []

    def fit(self, traces: Iterable[Trace]) -> "DataDrivenRepairer":
        self.frequencies = Counter(traces)
        if not self.frequencies:
            raise ValueError("É necessário ao menos um trace não sinalizado.")
        frequent = [
            trace
            for trace, frequency in self.frequencies.items()
            if frequency >= self.min_frequency
        ]
        candidates = frequent or list(self.frequencies)
        self.variants = sorted(
            candidates,
            key=lambda trace: (-self.frequencies[trace], len(trace), trace),
        )
        return self

    @property
    def prototype_count(self) -> int:
        return len(self.variants)

    def repair(self, trace: Trace) -> RepairResult:
        if not self.variants:
            raise RuntimeError("O reparador precisa ser ajustado antes do uso.")

        candidates = self.variants
        if self.max_candidates and len(candidates) > self.max_candidates:
            candidates = sorted(
                candidates,
                key=lambda item: (
                    abs(len(item) - len(trace)),
                    -self.frequencies[item],
                    item,
                ),
            )[: self.max_candidates]

        best = min(
            candidates,
            key=lambda candidate: (
                damerau_levenshtein(trace, candidate),
                -self.frequencies[candidate],
                candidate,
            ),
        )
        best_cost = damerau_levenshtein(trace, best)
        if self.max_edit_cost is not None and best_cost > self.max_edit_cost:
            return RepairResult(
                original_trace=trace,
                repaired_trace=trace,
                edit_cost=0,
                prototype_frequency=self.frequencies[best],
                repair_applied=False,
            )
        return RepairResult(
            original_trace=trace,
            repaired_trace=best,
            edit_cost=best_cost,
            prototype_frequency=self.frequencies[best],
            repair_applied=best != trace,
        )

    def repair_if_flagged(self, traces: list[Trace], flags: list[int]) -> list[RepairResult]:
        if len(traces) != len(flags):
            raise ValueError("traces e flags devem ter o mesmo tamanho.")
        output = []
        for trace, flag in zip(traces, flags):
            if flag:
                output.append(self.repair(trace))
            else:
                output.append(
                    RepairResult(trace, trace, 0, self.frequencies.get(trace, 0), False)
                )
        return output
