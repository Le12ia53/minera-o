from __future__ import annotations

from collections import Counter
import math

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


Trace = tuple[str, ...]


def normative_variants() -> set[Trace]:
    """Retorna as seis linearizações do BPMN apenas para avaliação sintética.

    Este conjunto não é utilizado por nenhum detector. Ele funciona como
    oráculo experimental para conferir a geração e avaliar o resultado final
    no cenário sintético controlado.
    """

    variants: set[Trace] = set()
    for choice in ("d", "e"):
        variants.add(("a", "b", "c", choice, "f"))
        variants.add(("a", "c", "b", choice, "f"))
        variants.add(("a", "c", choice, "b", "f"))
    return variants


def _validate_traces(traces: list[Trace], minimum: int = 2) -> None:
    if len(traces) < minimum:
        raise ValueError(f"São necessários ao menos {minimum} traces.")
    if any(not trace for trace in traces):
        raise ValueError("Traces vazios não são suportados.")


class TraceVectorizer:
    """Codifica traces sem consultar modelo normativo ou rótulos.

    A representação combina comprimento, contagens de atividades e contagens
    de sucessões diretas (bigramas). O vocabulário é aprendido no próprio log.
    """

    def fit(self, traces: list[Trace]) -> "TraceVectorizer":
        _validate_traces(traces)
        self.activities = sorted({activity for trace in traces for activity in trace})
        self.bigrams = sorted({edge for trace in traces for edge in zip(trace, trace[1:])})
        return self

    def transform(self, traces: list[Trace]) -> np.ndarray:
        if not hasattr(self, "activities"):
            raise RuntimeError("O vetorizador precisa ser ajustado antes do uso.")
        rows: list[list[float]] = []
        known_activities = set(self.activities)
        known_bigrams = set(self.bigrams)
        for trace in traces:
            activity_counts = Counter(trace)
            bigram_counts = Counter(zip(trace, trace[1:]))
            row: list[float] = [float(len(trace)), float(len(set(trace)))]
            row += [float(activity_counts[item]) for item in self.activities]
            row += [
                float(sum(v for k, v in activity_counts.items() if k not in known_activities))
            ]
            row += [float(bigram_counts[item]) for item in self.bigrams]
            row += [float(sum(v for k, v in bigram_counts.items() if k not in known_bigrams))]
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def fit_transform(self, traces: list[Trace]) -> np.ndarray:
        return self.fit(traces).transform(traces)


class TransitionRarityScorer:
    """Pontua traces pela surpresa estatística de suas transições.

    Frequências são estimadas no próprio conjunto observado com suavização
    aditiva. Uma transição rara aumenta o escore, mas não é automaticamente
    chamada de erro: o resultado serve para ranqueamento e investigação.
    """

    def __init__(self, alpha: float = 1.0, maximum_weight: float = 0.70):
        if alpha <= 0:
            raise ValueError("alpha deve ser positivo.")
        if not 0 <= maximum_weight <= 1:
            raise ValueError("maximum_weight deve estar entre 0 e 1.")
        self.alpha = alpha
        self.maximum_weight = maximum_weight

    def fit_score(self, traces: list[Trace]) -> np.ndarray:
        _validate_traces(traces)
        counts = Counter(edge for trace in traces for edge in zip(trace, trace[1:]))
        total = sum(counts.values())
        vocabulary = max(1, len(counts))
        denominator = total + self.alpha * vocabulary
        scores = []
        for trace in traces:
            edges = list(zip(trace, trace[1:]))
            if not edges:
                scores.append(0.0)
                continue
            surprises = [
                -math.log((counts[edge] + self.alpha) / denominator) for edge in edges
            ]
            mean_surprise = sum(surprises) / len(surprises)
            score = self.maximum_weight * max(surprises) + (
                1.0 - self.maximum_weight
            ) * mean_surprise
            scores.append(score)
        return np.asarray(scores, dtype=float)


class LocalOutlierFactorScorer:
    """Produz escores LOF no próprio log, sem catálogo de variantes válidas."""

    def __init__(self, n_neighbors: int = 20):
        if n_neighbors < 2:
            raise ValueError("n_neighbors deve ser pelo menos 2.")
        self.n_neighbors = n_neighbors

    def fit_score(self, traces: list[Trace]) -> np.ndarray:
        _validate_traces(traces, minimum=3)
        matrix = TraceVectorizer().fit_transform(traces)
        matrix = StandardScaler().fit_transform(matrix)
        neighbors = min(self.n_neighbors, len(traces) - 1)
        model = LocalOutlierFactor(n_neighbors=neighbors, contamination="auto")
        model.fit_predict(matrix)
        return -model.negative_outlier_factor_


class IsolationForestScorer:
    """Produz escores Isolation Forest sem usar rótulos ou modelo normativo."""

    def __init__(self, random_state: int = 42, n_estimators: int = 300):
        self.random_state = random_state
        self.n_estimators = n_estimators

    def fit_score(self, traces: list[Trace]) -> np.ndarray:
        _validate_traces(traces)
        matrix = TraceVectorizer().fit_transform(traces)
        matrix = StandardScaler().fit_transform(matrix)
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination="auto",
            max_samples="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(matrix)
        return -model.score_samples(matrix)


def percentile_ranks(scores: np.ndarray) -> np.ndarray:
    """Converte escores em posições normalizadas; maior continua mais anômalo."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("scores deve ser um vetor não vazio.")
    if len(values) == 1:
        return np.ones(1, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float) / (len(values) - 1)
    return ranks


def consensus_score(score_sets: list[np.ndarray]) -> np.ndarray:
    """Combina detectores pela média de seus rankings, sem calibrar por rótulos."""

    if not score_sets:
        raise ValueError("É necessário fornecer ao menos um conjunto de escores.")
    lengths = {len(scores) for scores in score_sets}
    if len(lengths) != 1:
        raise ValueError("Todos os conjuntos de escores devem ter o mesmo tamanho.")
    return np.mean(np.vstack([percentile_ranks(scores) for scores in score_sets]), axis=0)


def top_fraction_flags(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Sinaliza a fração de maior escore como fila de inspeção.

    O parâmetro representa um orçamento operacional de revisão, não a alegação
    de que todos os casos sinalizados são erros verdadeiros.
    """

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("scores deve ser um vetor não vazio.")
    if not 0 < fraction < 1:
        raise ValueError("fraction deve estar entre 0 e 1.")
    count = max(1, round(len(values) * fraction))
    order = np.lexsort((np.arange(len(values)), -values))
    flags = np.zeros(len(values), dtype=int)
    flags[order[:count]] = 1
    return flags


def score_all_methods(
    traces: list[Trace], random_state: int = 42, n_neighbors: int = 20
) -> dict[str, np.ndarray]:
    """Executa os métodos não supervisionados e o consenso proposto."""

    components = {
        "raridade_transicoes": TransitionRarityScorer().fit_score(traces),
        "local_outlier_factor": LocalOutlierFactorScorer(n_neighbors).fit_score(traces),
        "isolation_forest": IsolationForestScorer(random_state).fit_score(traces),
    }
    return {
        **components,
        "consenso_ranks": consensus_score(list(components.values())),
    }
