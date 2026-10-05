from __future__ import annotations

import random
from dataclasses import replace
from datetime import datetime, timedelta

from .log_io import Event, group_events


ANOMALY_TYPES = ("missing", "duplicate", "swap", "unknown")


def inject_noise(
    events: list[Event], noise_rate: float = 0.20, seed: int = 42
) -> tuple[list[Event], dict[str, tuple[int, str]]]:
    """Altera uma cópia do log e retorna rótulos por caso."""
    if not 0 < noise_rate < 1:
        raise ValueError("noise_rate deve estar entre 0 e 1.")
    grouped = group_events(events)
    case_ids = sorted(
        (case_id for case_id, case_events in grouped.items() if len(case_events) >= 3),
        key=_case_sort_key,
    )
    anomaly_count = max(len(ANOMALY_TYPES), round(len(case_ids) * noise_rate))
    if anomaly_count > len(case_ids):
        raise ValueError("Quantidade de anomalias maior que a quantidade de casos.")

    rng = random.Random(seed)
    selected = rng.sample(case_ids, anomaly_count)
    labels = {case_id: (0, "normal") for case_id in case_ids}

    for index, case_id in enumerate(selected):
        kind = ANOMALY_TYPES[index % len(ANOMALY_TYPES)]
        grouped[case_id] = _mutate(grouped[case_id], kind, rng)
        labels[case_id] = (1, kind)

    result = [event for case_id in case_ids for event in grouped[case_id]]
    return result, labels


def _mutate(events: list[Event], kind: str, rng: random.Random) -> list[Event]:
    items = list(events)
    if len(items) < 3:
        raise ValueError("Trace curto demais para receber os ruídos definidos.")

    if kind == "missing":
        del items[rng.randrange(1, len(items) - 1)]
    elif kind == "duplicate":
        pos = rng.randrange(1, len(items) - 1)
        duplicate = replace(
            items[pos],
            event_id=f"{items[pos].event_id}-DUP",
            timestamp=_shift_timestamp(items[pos].timestamp, microseconds=1),
        )
        items.insert(pos + 1, duplicate)
    elif kind == "swap":
        # Troca o início com a atividade seguinte para garantir violação normativa.
        first, second = items[0], items[1]
        items[0] = replace(first, activity=second.activity)
        items[1] = replace(second, activity=first.activity)
    elif kind == "unknown":
        pos = rng.randrange(1, len(items) - 1)
        unknown = replace(
            items[pos],
            event_id=f"{items[pos].event_id}-UNK",
            activity="__UNKNOWN__",
            timestamp=_shift_timestamp(items[pos].timestamp, microseconds=1),
        )
        items.insert(pos + 1, unknown)
    else:
        raise ValueError(f"Tipo de anomalia desconhecido: {kind}")
    return items


def _shift_timestamp(value: str, microseconds: int) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (parsed + timedelta(microseconds=microseconds)).isoformat(sep=" ")
    except ValueError:
        return value + f".{microseconds:06d}"


def _case_sort_key(case_id: str):
    try:
        return (0, int(case_id))
    except ValueError:
        return (1, case_id)
