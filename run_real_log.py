from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from src.detectors import (
    IsolationForestScorer,
    TransitionRarityScorer,
    consensus_score,
    top_fraction_flags,
)
from src.evaluation import binary_metrics, ranking_metrics, repair_metrics, write_csv
from src.log_io import group_traces, read_csv_events, read_xes_events, write_events
from src.noise import inject_noise
from src.pipeline import run_model_free_pipeline


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecção model-free e reparação data-driven no BPI Challenge 2012"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/real/BPI_Challenge_2012.xes.gz",
    )
    parser.add_argument("--top-rate", type=float, default=0.05)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--noise-rate", type=float, default=0.20)
    parser.add_argument(
        "--alert-rate",
        type=float,
        default=None,
        help="Fração da amostra controlada enviada à inspeção.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_events(path: Path):
    lower_name = path.name.lower()
    return read_xes_events(path) if lower_name.endswith((".xes", ".xes.gz")) else read_csv_events(path)


def _sample_case_ids(
    traces: dict[str, tuple[str, ...]], sample_size: int, seed: int
) -> list[str]:
    eligible = sorted(case_id for case_id, trace in traces.items() if len(trace) >= 3)
    if not eligible:
        raise ValueError("O log real não contém traces elegíveis.")
    rng = random.Random(seed)
    return sorted(rng.sample(eligible, min(sample_size, len(eligible))))


def main() -> None:
    args = parse_args()
    alert_rate = args.noise_rate if args.alert_rate is None else args.alert_rate
    for name, value in (
        ("top-rate", args.top_rate),
        ("noise-rate", args.noise_rate),
        ("alert-rate", alert_rate),
    ):
        if not 0 < value < 1:
            raise ValueError(f"{name} deve estar entre 0 e 1.")
    if args.sample_size < 20:
        raise ValueError("sample-size deve ser pelo menos 20.")
    if not args.input.exists():
        raise FileNotFoundError(
            f"Log real não encontrado: {args.input}. Consulte data/real/README.md."
        )

    events = _read_events(args.input)
    traces_by_case = group_traces(events)
    all_case_ids = sorted(traces_by_case)
    all_traces = [traces_by_case[case_id] for case_id in all_case_ids]
    activities = sorted({event.activity for event in events})
    variants = Counter(all_traces)

    # Análise nativa escalável: somente o próprio log é usado. Não há modelo
    # normativo nem rótulo; "flag" significa candidato à inspeção.
    native_transition = TransitionRarityScorer().fit_score(all_traces)
    native_isolation = IsolationForestScorer(args.seed).fit_score(all_traces)
    native_consensus = consensus_score([native_transition, native_isolation])
    native_flags = top_fraction_flags(native_consensus, args.top_rate)
    ranking_order = sorted(
        range(len(all_case_ids)),
        key=lambda index: (-native_consensus[index], all_case_ids[index]),
    )
    ranking_rows = [
        {
            "rank": rank,
            "case_id": all_case_ids[index],
            "trace": " -> ".join(all_traces[index]),
            "trace_length": len(all_traces[index]),
            "score_raridade_transicoes": float(native_transition[index]),
            "score_isolation_forest": float(native_isolation[index]),
            "score_consenso_ranks": float(native_consensus[index]),
            "candidato_inspecao_top_rate": int(native_flags[index]),
            "interpretacao": "candidato; nao e rotulo de erro",
        }
        for rank, index in enumerate(ranking_order, start=1)
    ]

    # Avaliação controlada em casos reais. A estratégia recebe apenas a versão
    # observada já corrompida; cópias limpas e rótulos ficam isolados até a
    # etapa posterior de avaliação.
    sampled_ids = set(_sample_case_ids(traces_by_case, args.sample_size, args.seed))
    sample_events = [event for event in events if event.case_id in sampled_ids]
    noisy_events, labels = inject_noise(sample_events, args.noise_rate, args.seed)
    noisy_sample = group_traces(noisy_events)
    case_ids = sorted(noisy_sample)
    observed = [noisy_sample[case_id] for case_id in case_ids]
    output = run_model_free_pipeline(
        observed,
        alert_rate=alert_rate,
        random_state=args.seed,
        n_neighbors=250,
        max_repair_candidates=250,
        min_prototype_frequency=2,
    )

    # Somente a partir daqui entram os dados reservados à avaliação.
    clean_sample = group_traces(sample_events)
    originals = [clean_sample[case_id] for case_id in case_ids]
    truth = [labels[case_id][0] for case_id in case_ids]
    metrics_rows = []
    repair_rows = []
    for method in output.scores:
        prediction = output.flags[method].tolist()
        repaired = [item.repaired_trace for item in output.repairs[method]]
        metrics_rows.append(
            {
                "metodo": method,
                **binary_metrics(truth, prediction),
                **ranking_metrics(truth, output.scores[method].tolist()),
                "seed": args.seed,
                "noise_rate": args.noise_rate,
                "alert_rate": alert_rate,
                "sample_cases": len(case_ids),
            }
        )
        repair_rows.append(
            {
                "metodo": method,
                **repair_metrics(originals, observed, repaired, truth),
                "prototipos_aprendidos": output.prototype_counts[method],
                "seed": args.seed,
                "noise_rate": args.noise_rate,
                "alert_rate": alert_rate,
                "sample_cases": len(case_ids),
            }
        )

    prediction_rows = []
    for index, case_id in enumerate(case_ids):
        row = {
            "case_id": case_id,
            "is_anomaly": truth[index],
            "anomaly_type": labels[case_id][1],
            "trace_original_oraculo": " -> ".join(originals[index]),
            "trace_observado": " -> ".join(observed[index]),
        }
        for method in output.scores:
            result = output.repairs[method][index]
            row[f"score_{method}"] = float(output.scores[method][index])
            row[f"pred_{method}"] = int(output.flags[method][index])
            row[f"trace_reparado_{method}"] = " -> ".join(result.repaired_trace)
            row[f"custo_reparo_{method}"] = result.edit_cost
            row[f"frequencia_prototipo_{method}"] = result.prototype_frequency
            row[f"reparo_aplicado_{method}"] = int(result.repair_applied)
            row[f"recuperacao_exata_{method}"] = int(
                result.repaired_trace == originals[index]
            )
        prediction_rows.append(row)

    type_rows = []
    anomaly_types = sorted({kind for label, kind in labels.values() if label})
    for method in output.scores:
        predictions = output.flags[method]
        for anomaly_type in anomaly_types:
            positions = [
                index
                for index, case_id in enumerate(case_ids)
                if labels[case_id][1] == anomaly_type
            ]
            detected = sum(int(predictions[index]) for index in positions)
            recovered = sum(
                output.repairs[method][index].repaired_trace == originals[index]
                for index in positions
            )
            type_rows.append(
                {
                    "metodo": method,
                    "anomaly_type": anomaly_type,
                    "total": len(positions),
                    "detected": detected,
                    "detection_recall": detected / len(positions) if positions else 0.0,
                    "exactly_recovered": recovered,
                    "exact_recovery_rate": recovered / len(positions) if positions else 0.0,
                }
            )

    stats = [
        {
            "dataset": "BPI Challenge 2012",
            "cases": len(traces_by_case),
            "events": len(events),
            "activities": len(activities),
            "variants": len(variants),
            "min_trace_length": min(map(len, all_traces)),
            "max_trace_length": max(map(len, all_traces)),
            "sample_cases": len(case_ids),
            "sample_anomalies": sum(truth),
            "native_candidates_top_rate": int(native_flags.sum()),
            "top_rate": args.top_rate,
            "seed": args.seed,
        }
    ]

    write_csv(ROOT / "results/estatisticas_log_real.csv", stats)
    write_csv(ROOT / "results/ranking_log_real.csv", ranking_rows)
    write_csv(ROOT / "results/metricas_real_controlado.csv", metrics_rows)
    write_csv(ROOT / "results/metricas_reparo_real_controlado.csv", repair_rows)
    write_csv(ROOT / "results/metricas_real_por_tipo_ruido.csv", type_rows)
    write_csv(ROOT / "results/predicoes_real_controlado.csv", prediction_rows)
    write_events(ROOT / "data/real/amostra_bpi2012_ruidosa.csv", noisy_events, labels)
    write_csv(
        ROOT / "data/real/rotulos_amostra_bpi2012.csv",
        [
            {
                "case_id": case_id,
                "is_anomaly": labels[case_id][0],
                "anomaly_type": labels[case_id][1],
            }
            for case_id in case_ids
        ],
    )

    print(
        f"BPI 2012: {len(traces_by_case)} casos | {len(events)} eventos | "
        f"{len(activities)} atividades | {len(variants)} variantes"
    )
    print(
        f"Experimento controlado: {len(case_ids)} casos | {sum(truth)} anomalias | "
        f"semente {args.seed}"
    )
    print("Detecção e reparação executadas sem modelo normativo e sem cópia limpa.")
    for row in metrics_rows:
        print(
            f"{row['metodo']}: precisão={row['precision']:.3f}, "
            f"recall={row['recall']:.3f}, F1={row['f1']:.3f}, "
            f"AP={row['average_precision']:.3f}"
        )
    for row in repair_rows:
        print(
            f"reparo/{row['metodo']}: recuperação exata="
            f"{row['exact_recovery_rate']:.3f}, melhora={row['improvement_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
