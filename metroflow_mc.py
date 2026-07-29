from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
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


def stable_seed_offset(mode: str, omega: float, r: float) -> int:
    token = f"{mode}|{omega:.12g}|{r:.12g}".encode("ascii")
    return zlib.crc32(token) % 1_000_000


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


def baseline_row(window: str, dmp_threshold: float, result: dict) -> dict:
    return {
        "window": window,
        "lambda0_dmp": dmp_threshold,
        "lambda0_mc": result["lambda_mc"],
        "lambda0_mc_ci95": result["lambda_mc_ci95"],
        "mc_runs": result["runs"],
        "dmp_mode_ipr": result["dmp_mode_ipr"],
        "lambda_scan": result["lambda_scan"],
        "growth_scan": result["growth_scan"],
    }


def result_key(row: dict) -> tuple[str, str, float, float]:
    return (
        row["window"],
        row["mode"],
        float(row["omega"]),
        float(row["r"]),
    )


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
    parser.add_argument(
        "--refinement-runs",
        type=int,
        default=5000,
        help=(
            "Runs used to recompute every preliminary DMP/MC point-sign "
            "mismatch; set to 0 to disable refinement."
        ),
    )
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
        default=",".join(
            str(value) for value in np.geomspace(10.0**-0.6, 10.0**0.6, 5)
        ),
    )
    parser.add_argument(
        "--omega-ratios",
        default=",".join(
            str(value) for value in np.geomspace(10.0**-0.5, 10.0**0.5, 5)
        ),
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
    contexts: dict[str, dict] = {}
    window_indices_by_name: dict[str, int] = {}
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
        contexts[window] = context
        window_indices_by_name[window] = window_index
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
            baseline_row(window, context["lambda0"], baseline_mc)
        )

        cache: dict[tuple[str, float, float], tuple[float, dict]] = {}

        def evaluate(mode: str, omega: float, r: float):
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
                    + stable_seed_offset(mode, omega, r),
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
            for r in primary_ratios:
                dmp_threshold, mc_result = evaluate(
                    "data-share",
                    args.omega,
                    float(r),
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
            for mode in MODES:
                for r in allocation_ratios:
                    dmp_threshold, mc_result = evaluate(
                        mode,
                        args.omega,
                        float(r),
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
            for omega in OMEGAS:
                for r in omega_ratios:
                    dmp_threshold, mc_result = evaluate(
                        "data-share",
                        omega,
                        float(r),
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

    all_rows = primary_rows + allocation_rows + omega_rows
    refinement_targets = {
        result_key(row)
        for row in all_rows
        if float(row["dmp_gain_pct"]) * float(row["mc_gain_pct"]) < 0.0
    }
    replacements: dict[tuple[str, str, float, float], dict] = {}
    baseline_replacements: dict[str, dict] = {}
    if args.refinement_runs > args.runs:
        for window in sorted({key[0] for key in refinement_targets}):
            context = contexts[window]
            window_index = window_indices_by_name[window]
            zeros = np.zeros_like(context["q_c"])
            baseline_matrix = build_B(
                context["k_c"],
                context["k_n"],
                zeros,
                zeros,
            )
            refined_baseline = estimate_threshold(
                baseline_matrix,
                context["lambda0"],
                args.mu,
                args.seed + window_index * 100000,
                args.refinement_runs,
                args.steps,
                args.fit_start,
                args.fit_end,
                args.initial_count,
                lambda_ratios,
                args.bootstraps,
            )
            baseline_replacements[window] = baseline_row(
                window,
                context["lambda0"],
                refined_baseline,
            )
            for _, mode, omega, r in sorted(
                (key for key in refinement_targets if key[0] == window),
                key=lambda key: (key[1], key[2], key[3]),
            ):
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
                refined_coupled = estimate_threshold(
                    matrix,
                    dmp_threshold,
                    args.mu,
                    args.seed
                    + window_index * 100000
                    + 10000
                    + stable_seed_offset(mode, omega, r),
                    args.refinement_runs,
                    args.steps,
                    args.fit_start,
                    args.fit_end,
                    args.initial_count,
                    lambda_ratios,
                    args.bootstraps,
                )
                key = (window, mode, omega, r)
                replacements[key] = gain_row(
                    window,
                    mode,
                    omega,
                    r,
                    context["lambda0"],
                    dmp_threshold,
                    refined_baseline,
                    refined_coupled,
                )

    def apply_replacements(rows: list[dict]) -> list[dict]:
        return [replacements.get(result_key(row), row) for row in rows]

    primary_rows = apply_replacements(primary_rows)
    allocation_rows = apply_replacements(allocation_rows)
    omega_rows = apply_replacements(omega_rows)
    baseline_rows.extend(baseline_replacements.values())
    baseline_rows.sort(key=lambda row: (row["window"], int(row["mc_runs"])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [args.output_dir / "mc_baselines.csv"]
    write_csv(output_paths[-1], baseline_rows)
    if primary_rows:
        output_paths.append(args.output_dir / "primary_scan_mc.csv")
        write_csv(output_paths[-1], primary_rows)
    if allocation_rows:
        output_paths.append(args.output_dir / "allocation_rule_mc.csv")
        write_csv(output_paths[-1], allocation_rows)
    if omega_rows:
        output_paths.append(args.output_dir / "omega_sensitivity_mc.csv")
        write_csv(output_paths[-1], omega_rows)
    settings = {
        "runs": args.runs,
        "refinement_runs": args.refinement_runs,
        "refinement_rule": (
            "Every preliminary DMP/MC point-sign mismatch is recomputed "
            "with refinement_runs; no point-specific seed selection."
        ),
        "refined_unique_configurations": len(replacements),
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
        "coupled_seed_rule": (
            "base + 100000*window_index + 10000 + "
            "crc32(mode|omega|r) mod 1000000"
        ),
    }
    (args.output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "settings": settings,
                "outputs": [str(path) for path in output_paths],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
