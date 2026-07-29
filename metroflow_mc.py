from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigs

from metroflow_dmp import (
    MODES,
    OMEGAS,
    WINDOWS,
    build_B,
    make_context,
    matched_thetas,
    read_and_audit,
    read_calendar,
    read_station_info,
    threshold_gain,
)
from dmp_core import dmp_nonbacktracking_matrix


HERE = Path(__file__).resolve().parent


def dmp_node_weights(matrix, threshold: float) -> np.ndarray:
    operator, _ = dmp_nonbacktracking_matrix(matrix, threshold, 1.0)
    _, vectors = eigs(operator, k=1, which="LR")
    edge_weights = np.abs(vectors[:, 0])
    coo = matrix.tocoo()
    targets = coo.row[coo.data > 1e-12].astype(int)
    node_weights = np.bincount(
        targets,
        weights=edge_weights,
        minlength=matrix.shape[0],
    ).astype(float)
    node_weights /= node_weights.sum()
    return node_weights


def simulate_counts(
    matrix,
    lam: float,
    mu: float,
    node_weights: np.ndarray,
    seed: int,
    runs: int,
    steps: int,
    initial_count: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b_csc = matrix.tocsc()
    n = b_csc.shape[0]
    candidates = np.flatnonzero(node_weights > 0)
    probabilities = node_weights[candidates]
    probabilities /= probabilities.sum()
    counts = np.empty((runs, steps + 1), dtype=float)
    for run in range(runs):
        infected = np.zeros(n, dtype=bool)
        initial = rng.choice(
            candidates,
            size=initial_count,
            replace=False,
            p=probabilities,
        )
        infected[initial] = True
        counts[run, 0] = initial_count
        for step in range(1, steps + 1):
            newly = np.zeros(n, dtype=bool)
            for source in np.flatnonzero(infected):
                left, right = b_csc.indptr[source], b_csc.indptr[source + 1]
                targets = b_csc.indices[left:right]
                transmission = np.clip(
                    lam * b_csc.data[left:right],
                    0.0,
                    1.0,
                )
                newly[targets] |= rng.random(len(targets)) < transmission
            newly &= ~infected
            recovered = infected & (rng.random(n) < mu)
            infected = (infected & ~recovered) | newly
            counts[run, step] = infected.sum()
    return counts


def ensemble_slope(
    counts: np.ndarray,
    fit_start: int,
    fit_end: int,
) -> float:
    mean_counts = counts.mean(axis=0)
    times = np.arange(fit_start, fit_end + 1, dtype=float)
    values = np.maximum(mean_counts[fit_start : fit_end + 1], 1e-12)
    return float(np.polyfit(times, np.log(values), 1)[0])


def interpolate_threshold(
    lambdas: np.ndarray,
    slopes: np.ndarray,
) -> float:
    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    slopes = slopes[order]
    for index in range(len(slopes) - 1):
        left = slopes[index]
        right = slopes[index + 1]
        if left <= 0.0 <= right:
            denominator = right - left
            if abs(denominator) < 1e-12:
                return float(0.5 * (lambdas[index] + lambdas[index + 1]))
            return float(
                lambdas[index]
                - left
                * (lambdas[index + 1] - lambdas[index])
                / denominator
            )
    return float(lambdas[int(np.argmin(np.abs(slopes)))])


def estimate_threshold(
    matrix,
    dmp_threshold: float,
    mu: float,
    seed: int,
    runs: int,
    steps: int,
    fit_start: int,
    fit_end: int,
    initial_count: int,
    lambda_ratios: np.ndarray,
    bootstraps: int,
) -> dict:
    node_weights = dmp_node_weights(matrix, dmp_threshold)
    lambdas = dmp_threshold * lambda_ratios
    count_sets = []
    slopes = []
    for index, lam in enumerate(lambdas):
        counts = simulate_counts(
            matrix,
            float(lam),
            mu,
            node_weights,
            seed + index * 10007,
            runs,
            steps,
            initial_count,
        )
        count_sets.append(counts)
        slopes.append(ensemble_slope(counts, fit_start, fit_end))
    slopes_array = np.asarray(slopes, dtype=float)
    estimate = interpolate_threshold(lambdas, slopes_array)

    bootstrap_rng = np.random.default_rng(seed + 900001)
    bootstrap_estimates = np.empty(bootstraps, dtype=float)
    for bootstrap in range(bootstraps):
        sampled_slopes = []
        for counts in count_sets:
            indices = bootstrap_rng.integers(0, runs, size=runs)
            sampled_slopes.append(
                ensemble_slope(counts[indices], fit_start, fit_end)
            )
        bootstrap_estimates[bootstrap] = interpolate_threshold(
            lambdas,
            np.asarray(sampled_slopes),
        )
    lower, upper = np.percentile(bootstrap_estimates, [2.5, 97.5])
    return {
        "lambda_mc": estimate,
        "lambda_mc_ci95": 0.5 * float(upper - lower),
        "lambda_scan": ";".join(f"{value:.8g}" for value in lambdas),
        "growth_scan": ";".join(f"{value:.8g}" for value in slopes_array),
        "bootstrap_estimates": bootstrap_estimates,
        "dmp_mode_ipr": float(np.sum(node_weights**2)),
        "runs": runs,
    }


def gain_row(
    window: str,
    mode: str,
    omega: float,
    r: float,
    dmp_baseline: float,
    dmp_coupled: float,
    mc_baseline: dict,
    mc_coupled: dict,
) -> dict:
    dmp_gain = 100.0 * (dmp_coupled / dmp_baseline - 1.0)
    mc_gain = 100.0 * (
        mc_coupled["lambda_mc"] / mc_baseline["lambda_mc"] - 1.0
    )
    bootstrap_gain = 100.0 * (
        mc_coupled["bootstrap_estimates"]
        / mc_baseline["bootstrap_estimates"]
        - 1.0
    )
    lower, upper = np.percentile(bootstrap_gain, [2.5, 97.5])
    return {
        "window": window,
        "mode": mode,
        "omega": omega,
        "r": r,
        "lambda0_dmp": dmp_baseline,
        "lambda_c_dmp": dmp_coupled,
        "dmp_gain_pct": dmp_gain,
        "lambda0_mc": mc_baseline["lambda_mc"],
        "lambda0_mc_ci95": mc_baseline["lambda_mc_ci95"],
        "lambda_c_mc": mc_coupled["lambda_mc"],
        "lambda_c_mc_ci95": mc_coupled["lambda_mc_ci95"],
        "mc_gain_pct": mc_gain,
        "mc_gain_ci95_pct": 0.5 * float(upper - lower),
        "mc_gain_ci95_lower": float(lower),
        "mc_gain_ci95_upper": float(upper),
        "mc_runs": mc_coupled["runs"],
        "dmp_mode_ipr": mc_coupled["dmp_mode_ipr"],
        "lambda_scan": mc_coupled["lambda_scan"],
        "growth_scan": mc_coupled["growth_scan"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dominant-mode early-growth MC validation for MetroFlow."
    )
    parser.add_argument("--metroflow-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "metroflow_mc",
    )
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--fit-start", type=int, default=1)
    parser.add_argument("--fit-end", type=int, default=8)
    parser.add_argument("--initial-count", type=int, default=3)
    parser.add_argument("--bootstraps", type=int, default=200)
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument(
        "--lambda-ratios",
        default="0.85,0.90,0.95,1.00,1.05,1.10,1.15",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "primary", "allocation", "omega"),
        default="all",
    )
    parser.add_argument(
        "--window-indices",
        default="0,1,2,3,4",
        help="Comma-separated zero-based MetroFlow window indices.",
    )
    parser.add_argument(
        "--primary-ratios",
        default=",".join(str(value) for value in np.geomspace(0.1, 10.0, 21)),
    )
    parser.add_argument(
        "--allocation-ratios",
        default=",".join(str(value) for value in np.geomspace(0.1, 10.0, 5)),
    )
    parser.add_argument(
        "--omega-ratios",
        default="0.31622776601683794,0.5623413251903491,1.0,"
        "3.1622776601683795,10.0",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    lambda_ratios = np.asarray(
        [float(value) for value in args.lambda_ratios.split(",")],
        dtype=float,
    )
    primary_ratios = np.asarray(
        [float(value) for value in args.primary_ratios.split(",")],
        dtype=float,
    )
    allocation_ratios = np.asarray(
        [float(value) for value in args.allocation_ratios.split(",")],
        dtype=float,
    )
    omega_ratios = np.asarray(
        [float(value) for value in args.omega_ratios.split(",")],
        dtype=float,
    )
    window_indices = {
        int(value) for value in args.window_indices.split(",") if value.strip()
    }

    station_ids, _, neighbors, _ = read_station_info(
        args.metroflow_dir / "stationInfo.csv"
    )
    calendar = read_calendar(args.metroflow_dir / "MetaData" / "workday_calendar.csv")
    commute, noncommute, _ = read_and_audit(
        args.metroflow_dir,
        station_ids,
        calendar,
    )
    degree = np.asarray([len(adjacent) for adjacent in neighbors], dtype=float)

    baseline_rows = []
    primary_rows = []
    allocation_rows = []
    omega_rows = []
    for window_index, (window, _, _) in enumerate(WINDOWS):
        if window_index not in window_indices:
            continue
        context = make_context(
            neighbors,
            degree,
            commute[0, window_index],
            noncommute[0, window_index],
            args.mu,
        )
        zeros = np.zeros_like(context["q_c"])
        baseline_matrix = build_B(
            context["k_c"],
            context["k_n"],
            zeros,
            zeros,
        )
        baseline_mc = estimate_threshold(
            baseline_matrix,
            context["lambda0"],
            args.mu,
            args.seed + window_index * 100000,
            args.runs,
            args.steps,
            args.fit_start,
            args.fit_end,
            args.initial_count,
            lambda_ratios,
            args.bootstraps,
        )
        baseline_rows.append(
            {
                "window": window,
                "lambda0_dmp": context["lambda0"],
                "lambda0_mc": baseline_mc["lambda_mc"],
                "lambda0_mc_ci95": baseline_mc["lambda_mc_ci95"],
                "dmp_mode_ipr": baseline_mc["dmp_mode_ipr"],
                "lambda_scan": baseline_mc["lambda_scan"],
                "growth_scan": baseline_mc["growth_scan"],
            }
        )

        cache: dict[tuple[str, float, float], tuple[float, dict]] = {}

        def evaluate(mode: str, omega: float, r: float, config_index: int):
            key = (mode, float(omega), float(r))
            if key not in cache:
                theta_c_to_n, theta_n_to_c = matched_thetas(
                    context,
                    mode,
                    omega,
                    r,
                )
                matrix = build_B(
                    context["k_c"],
                    context["k_n"],
                    theta_c_to_n,
                    theta_n_to_c,
                )
                dmp_threshold, _ = threshold_gain(
                    context,
                    mode,
                    omega,
                    r,
                    args.mu,
                )
                result = estimate_threshold(
                    matrix,
                    dmp_threshold,
                    args.mu,
                    args.seed
                    + window_index * 100000
                    + 10000
                    + config_index * 1000,
                    args.runs,
                    args.steps,
                    args.fit_start,
                    args.fit_end,
                    args.initial_count,
                    lambda_ratios,
                    args.bootstraps,
                )
                cache[key] = (dmp_threshold, result)
            return cache[key]

        if args.scope in {"all", "primary"}:
            for r_index, r in enumerate(primary_ratios):
                dmp_threshold, mc_result = evaluate(
                    "data-share",
                    args.omega,
                    float(r),
                    100 + r_index,
                )
                primary_rows.append(
                    gain_row(
                        window,
                        "data-share",
                        args.omega,
                        float(r),
                        context["lambda0"],
                        dmp_threshold,
                        baseline_mc,
                        mc_result,
                    )
                )

        if args.scope in {"all", "allocation"}:
            for mode_index, mode in enumerate(MODES):
                for r_index, r in enumerate(allocation_ratios):
                    dmp_threshold, mc_result = evaluate(
                        mode,
                        args.omega,
                        float(r),
                        1000 + mode_index * 100 + r_index,
                    )
                    allocation_rows.append(
                        gain_row(
                            window,
                            mode,
                            args.omega,
                            float(r),
                            context["lambda0"],
                            dmp_threshold,
                            baseline_mc,
                            mc_result,
                        )
                    )

        if args.scope in {"all", "omega"}:
            for omega_index, omega in enumerate(OMEGAS):
                for r_index, r in enumerate(omega_ratios):
                    dmp_threshold, mc_result = evaluate(
                        "data-share",
                        omega,
                        float(r),
                        2000 + omega_index * 100 + r_index,
                    )
                    omega_rows.append(
                        gain_row(
                            window,
                            "data-share",
                            omega,
                            float(r),
                            context["lambda0"],
                            dmp_threshold,
                            baseline_mc,
                            mc_result,
                        )
                    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mc_baselines.csv", baseline_rows)
    if primary_rows:
        write_csv(args.output_dir / "primary_scan_mc.csv", primary_rows)
    if allocation_rows:
        write_csv(args.output_dir / "allocation_rule_mc.csv", allocation_rows)
    if omega_rows:
        write_csv(args.output_dir / "omega_sensitivity_mc.csv", omega_rows)
    settings = {
        "runs": args.runs,
        "steps": args.steps,
        "fit_window": [args.fit_start, args.fit_end],
        "initial_count": args.initial_count,
        "initialization": "DMP dominant nonbacktracking mode projected to nodes",
        "growth_score": "slope of log ensemble-mean infected count",
        "lambda_ratios": lambda_ratios.tolist(),
        "bootstraps": args.bootstraps,
        "primary_ratios": primary_ratios.tolist(),
        "allocation_ratios": allocation_ratios.tolist(),
        "omega_ratios": omega_ratios.tolist(),
        "scope": args.scope,
        "window_indices": sorted(window_indices),
        "seed": args.seed,
    }
    (args.output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "settings": settings,
                "outputs": [
                    str(args.output_dir / "primary_scan_mc.csv"),
                    str(args.output_dir / "allocation_rule_mc.csv"),
                    str(args.output_dir / "omega_sensitivity_mc.csv"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
