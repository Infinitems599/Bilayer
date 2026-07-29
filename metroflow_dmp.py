from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

from dmp_core import (
    EPS,
    allocate_theta,
    build_B,
    build_kernel,
    dmp_nonbacktracking_matrix,
    read_station_info,
    spectral_radius,
)


WINDOWS = [
    ("06:00--07:00", 6 * 60, 7 * 60),
    ("07:00--09:00", 7 * 60, 9 * 60),
    ("09:00--17:00", 9 * 60, 17 * 60),
    ("17:00--19:00", 17 * 60, 19 * 60),
    ("19:00--23:00", 19 * 60, 23 * 60),
]
GROUPS = ("all", "workday", "non-workday")
MODES = ("data-share", "uniform", "degree-weighted")
OMEGAS = (0.1, 0.3, 0.5, 0.7, 0.9)
FLOW_FIELDS = (
    "inFlow",
    "outFlow",
    "CinFlow",
    "HBOinFlow",
    "NHBinFlow",
    "CoutFlow",
    "HBOoutFlow",
    "NHBoutFlow",
)
PURPOSE_FIELDS = (
    "CinFlow",
    "HBOinFlow",
    "NHBinFlow",
    "CoutFlow",
    "HBOoutFlow",
    "NHBoutFlow",
)


def parse_time(value: str) -> int:
    text = value.strip().zfill(6)
    return int(text[:2]) * 60 + int(text[2:4])


def window_index(minute: int) -> int | None:
    for index, (_, start, end) in enumerate(WINDOWS):
        if start <= minute < end:
            return index
    return None


def read_calendar(path: Path) -> dict[str, bool]:
    calendar: dict[str, bool] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("The workday calendar has no header")
        date_key = next(name for name in reader.fieldnames if name.strip() == "date")
        flag_key = next(
            name for name in reader.fieldnames if name.strip() == "isWorday"
        )
        for row in reader:
            calendar[row[date_key].strip()] = bool(int(row[flag_key]))
    return calendar


def read_and_audit(
    metroflow_dir: Path,
    station_ids: list[int],
    calendar: dict[str, bool],
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(station_ids)
    id_to_index = {station: index for index, station in enumerate(station_ids)}
    dates = sorted(calendar)
    date_to_index = {date: index for index, date in enumerate(dates)}
    seen = np.zeros((len(dates), 102, n), dtype=bool)
    commute = np.zeros((len(GROUPS), len(WINDOWS), n), dtype=float)
    noncommute = np.zeros_like(commute)
    rows_by_group_window = np.zeros(
        (len(GROUPS), len(WINDOWS)),
        dtype=np.int64,
    )

    total_rows = 0
    duplicate_rows = 0
    unexpected_keys = 0
    missing_by_field = {field: 0 for field in FLOW_FIELDS}
    negative_by_field = {field: 0 for field in FLOW_FIELDS}
    zero_by_field = {field: 0 for field in FLOW_FIELDS}
    zero_purpose_rows = 0
    observed_dates: set[str] = set()
    observed_slots: set[int] = set()

    with (metroflow_dir / "metroData_InOutFlow.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        keys = {name.strip(): name for name in reader.fieldnames or []}
        required = {"date", "timeslot", "startTime", "station", *FLOW_FIELDS}
        missing_columns = sorted(required - set(keys))
        if missing_columns:
            raise ValueError(f"Missing MetroFlow columns: {missing_columns}")

        for row in reader:
            total_rows += 1
            date = row[keys["date"]].strip()
            global_slot = int(row[keys["timeslot"]])
            station = int(row[keys["station"]])
            minute = parse_time(row[keys["startTime"]])
            local_slot = (minute - 6 * 60) // 10
            observed_dates.add(date)
            observed_slots.add(local_slot)

            date_index = date_to_index.get(date)
            station_index = id_to_index.get(station)
            expected_slot = (
                date_index * 102 + local_slot if date_index is not None else None
            )
            if (
                date_index is None
                or station_index is None
                or not 0 <= local_slot < 102
                or global_slot != expected_slot
            ):
                unexpected_keys += 1
                continue
            if seen[date_index, local_slot, station_index]:
                duplicate_rows += 1
            else:
                seen[date_index, local_slot, station_index] = True

            values: dict[str, float] = {}
            for field in FLOW_FIELDS:
                raw = row[keys[field]].strip()
                if raw == "":
                    missing_by_field[field] += 1
                    values[field] = 0.0
                    continue
                value = float(raw)
                values[field] = value
                if value < 0:
                    negative_by_field[field] += 1
                if value == 0:
                    zero_by_field[field] += 1
            if sum(values[field] for field in PURPOSE_FIELDS) == 0:
                zero_purpose_rows += 1

            selected_window = window_index(minute)
            if selected_window is None:
                continue
            commuting_value = values["CinFlow"] + values["CoutFlow"]
            noncommuting_value = (
                values["HBOinFlow"]
                + values["HBOoutFlow"]
                + values["NHBinFlow"]
                + values["NHBoutFlow"]
            )
            for group_index in (0, 1 if calendar[date] else 2):
                commute[group_index, selected_window, station_index] += (
                    commuting_value
                )
                noncommute[group_index, selected_window, station_index] += (
                    noncommuting_value
                )
                rows_by_group_window[group_index, selected_window] += 1

    expected_records = len(dates) * 102 * n
    unique_records = int(seen.sum())
    audit = {
        "observation_start": min(observed_dates),
        "observation_end": max(observed_dates),
        "calendar_days": len(dates),
        "workdays": sum(calendar.values()),
        "non_workdays": len(calendar) - sum(calendar.values()),
        "stations": n,
        "daily_time_slots": len(observed_slots),
        "expected_records": expected_records,
        "actual_records": total_rows,
        "unique_station_time_records": unique_records,
        "duplicate_station_time_records": duplicate_rows,
        "missing_station_time_combinations": expected_records - unique_records,
        "unexpected_date_slot_station_keys": unexpected_keys,
        "missing_values_by_flow_field": missing_by_field,
        "negative_values_by_flow_field": negative_by_field,
        "zero_values_by_flow_field": zero_by_field,
        "zero_purpose_flow_records": zero_purpose_rows,
        "rows_by_group_window": {
            GROUPS[group]: {
                WINDOWS[window][0]: int(rows_by_group_window[group, window])
                for window in range(len(WINDOWS))
            }
            for group in range(len(GROUPS))
        },
    }
    return commute, noncommute, audit


def zero_neighbor_denominators(
    neighbors: list[list[int]],
    strength: np.ndarray,
) -> int:
    return sum(
        1
        for adjacent in neighbors
        if not adjacent or float(strength[adjacent].sum()) <= 0
    )


def dmp_radius_c1(matrix: sparse.spmatrix) -> float:
    coo = matrix.tocoo()
    mask = coo.data > EPS
    source = coo.col[mask].astype(int)
    target = coo.row[mask].astype(int)
    weight = coo.data[mask].astype(float)
    edge_count = len(source)
    if edge_count == 0:
        return 0.0

    incoming: list[list[int]] = [[] for _ in range(matrix.shape[0])]
    outgoing: list[list[int]] = [[] for _ in range(matrix.shape[0])]
    for edge, (src, dst) in enumerate(zip(source, target)):
        outgoing[src].append(edge)
        incoming[dst].append(edge)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for node in range(matrix.shape[0]):
        for new_edge in outgoing[node]:
            reverse_target = target[new_edge]
            for old_edge in incoming[node]:
                if source[old_edge] == reverse_target:
                    continue
                rows.append(new_edge)
                cols.append(old_edge)
                values.append(weight[old_edge])
    if not values:
        return 0.0
    operator = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(edge_count, edge_count),
        dtype=float,
    )
    eigenvalues = eigs(
        operator,
        k=1,
        which="LR",
        v0=np.full(edge_count, 1.0 / edge_count),
        return_eigenvectors=False,
        tol=1e-10,
        maxiter=20000,
    )
    return float(abs(eigenvalues[0]))


def matrix_radius(matrix: sparse.spmatrix) -> float:
    if matrix.shape[0] == 0 or matrix.nnz == 0:
        return 0.0
    return float(spectral_radius(matrix)[0])


def dmp_certificate_equivalent(
    matrix: sparse.spmatrix,
    lambda0: float,
    mu: float,
    layer_size: int,
) -> dict:
    """Evaluate Theorem 4.2 using its equivalent sparse M-matrix test."""
    operator, edge_sources = dmp_nonbacktracking_matrix(matrix, lambda0, 1.0)
    if operator.shape[0] == 0:
        return {
            "rho_A": 0.0,
            "rho_D": 0.0,
            "rho_full": 0.0,
            "certified": True,
        }
    alpha = np.flatnonzero(edge_sources < layer_size)
    beta = np.flatnonzero(edge_sources >= layer_size)
    order = np.r_[alpha, beta]
    ordered = operator[order, :][:, order].tocsr()
    split = alpha.size
    rho_a = matrix_radius(ordered[:split, :split])
    rho_d = matrix_radius(ordered[split:, split:])
    rho_full = matrix_radius(ordered)
    return {
        "rho_A": rho_a,
        "rho_D": rho_d,
        "rho_full": rho_full,
        "certified": (
            rho_a < mu - 1e-10
            and rho_d < mu - 1e-10
            and rho_full < mu - 1e-10
        ),
    }


def make_context(
    neighbors: list[list[int]],
    degree: np.ndarray,
    commute: np.ndarray,
    noncommute: np.ndarray,
    mu: float,
) -> dict:
    total = commute + noncommute
    q_c = np.divide(commute, total, out=np.zeros_like(commute), where=total > 0)
    q_n = np.divide(
        noncommute,
        total,
        out=np.zeros_like(noncommute),
        where=total > 0,
    )
    k_c = build_kernel(neighbors, commute, q_c)
    k_n = build_kernel(neighbors, noncommute, q_n)
    rho_c, _ = spectral_radius(k_c)
    rho_n, _ = spectral_radius(k_n)
    high_layer = "commuting" if rho_c >= rho_n else "non-commuting"
    zeros = np.zeros_like(q_c)
    b0 = build_B(k_c, k_n, zeros, zeros)
    rho_h0 = dmp_radius_c1(b0)
    if rho_h0 <= EPS:
        raise ValueError("The uncoupled DMP operator has zero spectral radius")
    return {
        "q_c": q_c,
        "q_n": q_n,
        "k_c": k_c,
        "k_n": k_n,
        "degree": degree,
        "rho_c": float(rho_c),
        "rho_n": float(rho_n),
        "high_layer": high_layer,
        "lambda0": float(mu / rho_h0),
        "zero_total_activity_stations": int(np.count_nonzero(total <= 0)),
        "zero_commuting_neighbor_denominators": zero_neighbor_denominators(
            neighbors,
            commute,
        ),
        "zero_noncommuting_neighbor_denominators": zero_neighbor_denominators(
            neighbors,
            noncommute,
        ),
    }


def matched_thetas(
    context: dict,
    mode: str,
    omega: float,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean_high_to_low = omega * ratio / (1.0 + ratio)
    mean_reverse = omega / (1.0 + ratio)
    if context["high_layer"] == "commuting":
        mean_c_to_n, mean_n_to_c = mean_high_to_low, mean_reverse
        profile_c_to_n, profile_n_to_c = context["q_n"], context["q_c"]
    else:
        mean_n_to_c, mean_c_to_n = mean_high_to_low, mean_reverse
        profile_n_to_c, profile_c_to_n = context["q_c"], context["q_n"]

    if mode == "data-share":
        theta_c_to_n = allocate_theta(mean_c_to_n, profile_c_to_n)
        theta_n_to_c = allocate_theta(mean_n_to_c, profile_n_to_c)
    elif mode == "uniform":
        theta_c_to_n = np.full_like(profile_c_to_n, mean_c_to_n)
        theta_n_to_c = np.full_like(profile_n_to_c, mean_n_to_c)
    elif mode == "degree-weighted":
        theta_c_to_n = allocate_theta(mean_c_to_n, context["degree"])
        theta_n_to_c = allocate_theta(mean_n_to_c, context["degree"])
    else:
        raise ValueError(f"Unknown allocation mode: {mode}")
    return theta_c_to_n, theta_n_to_c


def threshold_gain(
    context: dict,
    mode: str,
    omega: float,
    ratio: float,
    mu: float,
) -> tuple[float, float]:
    theta_c_to_n, theta_n_to_c = matched_thetas(
        context,
        mode,
        omega,
        ratio,
    )
    matrix = build_B(
        context["k_c"],
        context["k_n"],
        theta_c_to_n,
        theta_n_to_c,
    )
    radius = dmp_radius_c1(matrix)
    threshold = mu / radius if radius > EPS else math.inf
    gain = 100.0 * (threshold / context["lambda0"] - 1.0)
    return float(threshold), float(gain)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MetroFlow data audit and finite-lambda DMP threshold scans."
    )
    parser.add_argument("--metroflow-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "metroflow_dmp",
    )
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--r-steps", type=int, default=21)
    parser.add_argument("--representative-r", type=float, default=4.0)
    args = parser.parse_args()

    station_ids, _, neighbors, physical_edges = read_station_info(
        args.metroflow_dir / "stationInfo.csv"
    )
    calendar = read_calendar(
        args.metroflow_dir / "MetaData" / "workday_calendar.csv"
    )
    commute, noncommute, audit = read_and_audit(
        args.metroflow_dir,
        station_ids,
        calendar,
    )
    audit["physical_edges"] = physical_edges
    degree = np.asarray([len(adjacent) for adjacent in neighbors], dtype=float)

    contexts: dict[tuple[str, str], dict] = {}
    context_rows: list[dict] = []
    for group_index, group in enumerate(GROUPS):
        for window_index_value, (window, _, _) in enumerate(WINDOWS):
            context = make_context(
                neighbors,
                degree,
                commute[group_index, window_index_value],
                noncommute[group_index, window_index_value],
                args.mu,
            )
            contexts[(group, window)] = context
            context_rows.append(
                {
                    "day_group": group,
                    "window": window,
                    "rho_commuting": context["rho_c"],
                    "rho_noncommuting": context["rho_n"],
                    "high_risk_layer": context["high_layer"],
                    "lambda0_dmp": context["lambda0"],
                    "zero_total_activity_stations": context[
                        "zero_total_activity_stations"
                    ],
                    "zero_commuting_neighbor_denominators": context[
                        "zero_commuting_neighbor_denominators"
                    ],
                    "zero_noncommuting_neighbor_denominators": context[
                        "zero_noncommuting_neighbor_denominators"
                    ],
                }
            )

    ratios = np.unique(
        np.r_[
            np.geomspace(0.1, 10.0, args.r_steps),
            args.representative_r,
        ]
    )
    allocation_rows: list[dict] = []
    omega_rows: list[dict] = []
    primary_rows: list[dict] = []
    for window, _, _ in WINDOWS:
        context = contexts[("all", window)]
        for ratio in ratios:
            theta_c_to_n, theta_n_to_c = matched_thetas(
                context,
                "data-share",
                args.omega,
                float(ratio),
            )
            matrix = build_B(
                context["k_c"],
                context["k_n"],
                theta_c_to_n,
                theta_n_to_c,
            )
            threshold, gain = threshold_gain(
                context,
                "data-share",
                args.omega,
                float(ratio),
                args.mu,
            )
            certificate = dmp_certificate_equivalent(
                matrix,
                context["lambda0"],
                args.mu,
                len(context["q_c"]),
            )
            primary_rows.append(
                {
                    "window": window,
                    "r": float(ratio),
                    "omega": args.omega,
                    "lambda0_dmp": context["lambda0"],
                    "lambda_c_dmp": threshold,
                    "dmp_gain_pct": gain,
                    "dmp_rho_A_ratio": certificate["rho_A"] / args.mu,
                    "dmp_rho_D_ratio": certificate["rho_D"] / args.mu,
                    "dmp_radius_at_lambda0_ratio": (
                        certificate["rho_full"] / args.mu
                    ),
                    "dmp_theorem_4_2_satisfied": int(
                        certificate["certified"]
                    ),
                }
            )
        for mode in MODES:
            for ratio in ratios:
                threshold, gain = threshold_gain(
                    context,
                    mode,
                    args.omega,
                    float(ratio),
                    args.mu,
                )
                allocation_rows.append(
                    {
                        "window": window,
                        "mode": mode,
                        "omega": args.omega,
                        "r": float(ratio),
                        "lambda0_dmp": context["lambda0"],
                        "lambda_c_dmp": threshold,
                        "gain_pct": gain,
                    }
                )
        for omega in OMEGAS:
            for ratio in ratios:
                threshold, gain = threshold_gain(
                    context,
                    "data-share",
                    omega,
                    float(ratio),
                    args.mu,
                )
                omega_rows.append(
                    {
                        "window": window,
                        "mode": "data-share",
                        "omega": omega,
                        "r": float(ratio),
                        "lambda0_dmp": context["lambda0"],
                        "lambda_c_dmp": threshold,
                        "gain_pct": gain,
                    }
                )

    day_group_rows: list[dict] = []
    for group in GROUPS:
        for window, _, _ in WINDOWS:
            context = contexts[(group, window)]
            threshold, gain = threshold_gain(
                context,
                "data-share",
                args.omega,
                args.representative_r,
                args.mu,
            )
            day_group_rows.append(
                {
                    "day_group": group,
                    "window": window,
                    "rho_commuting": context["rho_c"],
                    "rho_noncommuting": context["rho_n"],
                    "high_risk_layer": context["high_layer"],
                    "lambda0_dmp": context["lambda0"],
                    "lambda_c_dmp": threshold,
                    "gain_pct": gain,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "window_contexts.csv", context_rows)
    write_csv(args.output_dir / "primary_dmp_scan.csv", primary_rows)
    write_csv(args.output_dir / "allocation_rule_scan.csv", allocation_rows)
    write_csv(args.output_dir / "omega_sensitivity.csv", omega_rows)
    write_csv(args.output_dir / "day_group_comparison.csv", day_group_rows)
    print(
        json.dumps(
            {
                "audit": str(args.output_dir / "data_audit.json"),
                "primary_dmp_scan": str(
                    args.output_dir / "primary_dmp_scan.csv"
                ),
                "allocation_rule_scan": str(
                    args.output_dir / "allocation_rule_scan.csv"
                ),
                "omega_sensitivity": str(
                    args.output_dir / "omega_sensitivity.csv"
                ),
                "day_group_comparison": str(
                    args.output_dir / "day_group_comparison.csv"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
