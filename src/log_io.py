from __future__ import annotations

import csv
import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Event:
    event_id: str
    case_id: str
    activity: str
    timestamp: str


def read_csv_events(path: str | Path) -> list[Event]:
    """Lê CSV aceitando os nomes de colunas usados nos materiais da disciplina."""
    aliases = {
        "event_id": ("Event ID", "eventID", "event_id"),
        "case_id": ("Case ID", "caseID", "case_id"),
        "activity": ("Activity", "activity", "concept:name"),
        "timestamp": ("Timestamp", "timestamp", "time:timestamp"),
    }
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])

        def choose(name: str, required: bool = True) -> str | None:
            value = next((item for item in aliases[name] if item in fields), None)
            if required and value is None:
                raise ValueError(f"Coluna obrigatória ausente: {aliases[name]}")
            return value

        event_col = choose("event_id", required=False)
        case_col = choose("case_id")
        activity_col = choose("activity")
        time_col = choose("timestamp")
        events = []
        for index, row in enumerate(reader, start=1):
            events.append(
                Event(
                    event_id=str(row.get(event_col, index) if event_col else index),
                    case_id=str(row[case_col]),
                    activity=str(row[activity_col]),
                    timestamp=str(row[time_col]),
                )
            )
    return events


def _xes_value(element: ET.Element, key: str) -> str | None:
    for child in element:
        if child.attrib.get("key") == key:
            return child.attrib.get("value")
    return None


def read_xes_events(path: str | Path) -> list[Event]:
    """Lê os campos mínimos de um arquivo XES ou XES.GZ sem alterar o arquivo."""
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as handle:
        root = ET.parse(handle).getroot()
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0] + "}"
    result: list[Event] = []
    event_index = 0
    for trace in root.findall(f"{namespace}trace"):
        case_id = _xes_value(trace, "concept:name")
        if case_id is None:
            raise ValueError("Trace XES sem concept:name (Case ID).")
        for event in trace.findall(f"{namespace}event"):
            event_index += 1
            activity = _xes_value(event, "concept:name")
            timestamp = _xes_value(event, "time:timestamp")
            if activity is None or timestamp is None:
                raise ValueError("Evento XES sem atividade ou timestamp.")
            result.append(Event(str(event_index), case_id, activity, timestamp))
    return result


def group_traces(events: Iterable[Event]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        grouped[event.case_id].append(event)
    traces: dict[str, tuple[str, ...]] = {}
    for case_id, case_events in grouped.items():
        case_events.sort(key=lambda event: _timestamp_key(event.timestamp))
        traces[case_id] = tuple(event.activity for event in case_events)
    return traces


def group_events(events: Iterable[Event]) -> dict[str, list[Event]]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        grouped[event.case_id].append(event)
    for case_events in grouped.values():
        case_events.sort(key=lambda event: _timestamp_key(event.timestamp))
    return dict(grouped)


def write_events(
    path: str | Path,
    events: Iterable[Event],
    labels: dict[str, tuple[int, str]] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Event ID", "Case ID", "Activity", "Timestamp"]
    if labels is not None:
        fields += ["is_anomaly", "anomaly_type"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            row = {
                "Event ID": event.event_id,
                "Case ID": event.case_id,
                "Activity": event.activity,
                "Timestamp": event.timestamp,
            }
            if labels is not None:
                label, kind = labels[event.case_id]
                row.update(is_anomaly=label, anomaly_type=kind)
            writer.writerow(row)


def _timestamp_key(value: str):
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return value

