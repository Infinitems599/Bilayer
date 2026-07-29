from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs


EPS = 1e-12


def read_station_info(path: Path):
    station_ids: list[int] = []
    names: list[str] = []
    raw_neighbors: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["stationID"])
            station_ids.append(sid)
            names.append(row["name"])
            raw_neighbors[sid] = [int(x) for x in ast.literal_eval(row["neighbour"])]

    id_to_idx = {sid: i for i, sid in enumerate(station_ids)}
    neighbors = [set() for _ in station_ids]
    for sid, neighs in raw_neighbors.items():
        if sid not in id_to_idx:
            continue
        i = id_to_idx[sid]
        for nb in neighs:
            if nb not in id_to_idx:
                continue
            j = id_to_idx[nb]
            if i != j:
                neighbors[i].add(j)
                neighbors[j].add(i)

    edges = set()
    for i, ns in enumerate(neighbors):
        for j in ns:
            edges.add((min(i, j), max(i, j)))
    return station_ids, names, [sorted(s) for s in neighbors], len(edges)


def parse_hhmmss(value: str) -> int:
    value = value.strip().zfill(6)
    return int(value[:2]) * 60 + int(value[2:4])


def parse_windows(spec: str | None):
    if not spec or spec.lower() == "all":
        return None
    windows = []
    for item in spec.split(","):
        left, right = item.split("-")
        lh, lm = [int(x) for x in left.split(":")]
        rh, rm = [int(x) for x in right.split(":")]
        windows.append((lh * 60 + lm, rh * 60 + rm))
    return windows


def in_window(start_time: str, windows) -> bool:
    if windows is None:
        return True
    minute = parse_hhmmss(start_time)
    return any(left <= minute < right for left, right in windows)


def aggregate_inout(path: Path, id_to_idx: dict[int, int], windows):
    n = len(id_to_idx)
    commute = np.zeros(n, dtype=float)
    noncommute = np.zeros(n, dtype=float)
    total_rows = 0
    used_rows = 0
    dates: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if not in_window(row[" startTime"], windows):
                continue
            sid = int(row[" station"])
            if sid not in id_to_idx:
                continue
            i = id_to_idx[sid]
            dates.add(row["date"].strip())
            commute[i] += float(row[" CinFlow"]) + float(row[" CoutFlow"])
            noncommute[i] += (
                float(row[" HBOinFlow"])
                + float(row[" HBOoutFlow"])
                + float(row[" NHBinFlow"])
                + float(row[" NHBoutFlow"])
            )
            used_rows += 1

    return commute, noncommute, {
        "total_rows": total_rows,
        "used_rows": used_rows,
        "dates": len(dates),
        "total_commute_flow": float(commute.sum()),
        "total_noncommute_flow": float(noncommute.sum()),
    }


def build_kernel(neighbors: list[list[int]], target_strength: np.ndarray, source_participation: np.ndarray):
    rows = []
    cols = []
    vals = []
    for j, ns in enumerate(neighbors):
        denom = float(target_strength[ns].sum()) if ns else 0.0
        if denom <= 0:
            continue
        for i in ns:
            value = source_participation[j] * target_strength[i] / denom
            if value:
                rows.append(i)
                cols.append(j)
                vals.append(value)
    n = len(neighbors)
    return sparse.csc_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)


def spectral_radius(mat: sparse.spmatrix) -> tuple[float, np.ndarray]:
    if mat.shape[0] == 1:
        return float(abs(mat[0, 0])), np.ones(1)
    try:
        values, vectors = eigs(mat, k=1, which="LM", tol=1e-10, maxiter=10000)
        idx = int(np.argmax(np.abs(values)))
        rho = float(abs(values[idx]))
        vec = np.abs(np.real(vectors[:, idx]))
    except Exception:
        dense = mat.toarray()
        values, vectors = np.linalg.eig(dense)
        idx = int(np.argmax(np.abs(values)))
        rho = float(abs(values[idx]))
        vec = np.abs(np.real(vectors[:, idx]))
    vec = np.maximum(vec, EPS)
    vec = vec / vec.sum()
    return rho, vec


def dmp_edge_arrays(B: sparse.spmatrix, lam: float, contact_capacity: float):
    coo = B.tocoo()
    mask = coo.data > EPS
    src = coo.col[mask].astype(int)
    tgt = coo.row[mask].astype(int)
    prob = 1.0 - np.maximum(1.0 - lam * coo.data[mask], EPS) ** contact_capacity
    return src, tgt, prob.astype(float), B.shape[0]


def dmp_nonbacktracking_matrix(
    B: sparse.spmatrix,
    lam: float,
    contact_capacity: float,
):
    """Construct the weighted directed-edge DMP matrix."""
    src, tgt, prob, n = dmp_edge_arrays(B, lam, contact_capacity)
    edge_count = len(src)
    if edge_count == 0:
        return sparse.csr_matrix((0, 0), dtype=float), src

    incoming = [[] for _ in range(n)]
    outgoing = [[] for _ in range(n)]
    for edge, (source, target) in enumerate(zip(src, tgt)):
        incoming[target].append(edge)
        outgoing[source].append(edge)

    rows = []
    cols = []
    values = []
    for node in range(n):
        for new_edge in outgoing[node]:
            reverse_target = tgt[new_edge]
            for old_edge in incoming[node]:
                if src[old_edge] == reverse_target:
                    continue
                rows.append(new_edge)
                cols.append(old_edge)
                values.append(prob[old_edge])
    operator = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(edge_count, edge_count),
        dtype=float,
    )
    return operator, src


def dmp_nonbacktracking_radius(B: sparse.spmatrix, lam: float, contact_capacity: float) -> float:
    src, tgt, prob, n = dmp_edge_arrays(B, lam, contact_capacity)
    e_count = len(src)
    if e_count == 0:
        return 0.0
    incoming = [[] for _ in range(n)]
    outgoing = [[] for _ in range(n)]
    for e, (s, t) in enumerate(zip(src, tgt)):
        incoming[t].append(e)
        outgoing[s].append(e)
    rows = []
    cols = []
    data = []
    for node in range(n):
        if not incoming[node] or not outgoing[node]:
            continue
        for new_e in outgoing[node]:
            back_target = tgt[new_e]
            for old_e in incoming[node]:
                if src[old_e] == back_target:
                    continue
                rows.append(new_e)
                cols.append(old_e)
                data.append(prob[old_e])
    if not data:
        return 0.0
    H = sparse.csr_matrix((data, (rows, cols)), shape=(e_count, e_count))
    rho, _ = spectral_radius(H)
    return rho


def lambda_c_dmp(B: sparse.spmatrix, mu: float, contact_capacity: float) -> float:
    low = 0.0
    high = 0.08
    while high < 1.0 and dmp_nonbacktracking_radius(B, high, contact_capacity) < mu:
        high = min(1.0, high * 2.0)
    if dmp_nonbacktracking_radius(B, high, contact_capacity) < mu:
        return float(high)
    for _ in range(34):
        mid = 0.5 * (low + high)
        if dmp_nonbacktracking_radius(B, mid, contact_capacity) < mu:
            low = mid
        else:
            high = mid
    return float(high)


def allocate_theta(mean_theta: float, weights: np.ndarray, cap: float = 1.0) -> np.ndarray:
    n = len(weights)
    mean_theta = float(np.clip(mean_theta, 0.0, cap))
    if mean_theta <= 0:
        return np.zeros(n, dtype=float)
    if mean_theta >= cap:
        return np.full(n, cap, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.0)
    if float(w.sum()) <= 0:
        return np.full(n, mean_theta, dtype=float)

    target = mean_theta * n
    if target >= cap * np.count_nonzero(w > 0):
        return np.where(w > 0, cap, 0.0)

    lo = 0.0
    hi = cap / max(float(w.max()), EPS)
    while float(np.minimum(cap, hi * w).sum()) < target:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        values = np.minimum(cap, mid * w)
        if float(values.sum()) < target:
            lo = mid
        else:
            hi = mid
    return np.minimum(cap, hi * w)


def coupling_weights(mode: str, degree: np.ndarray, source_strength: np.ndarray, target_strength: np.ndarray) -> np.ndarray:
    if mode == "uniform":
        return np.ones_like(degree, dtype=float)
    if mode == "degree":
        return degree.astype(float)
    if mode == "inverse-degree":
        return 1.0 / np.maximum(degree.astype(float), 1.0)
    if mode == "activity":
        return np.asarray(source_strength, dtype=float)
    if mode == "target-activity":
        return np.asarray(target_strength, dtype=float)
    raise ValueError(f"Unknown coupling mode: {mode}")


def build_B(kc, kn, theta_c_to_n: np.ndarray, theta_n_to_c: np.ndarray):
    keep_c = sparse.diags(1.0 - theta_c_to_n, format="csc")
    keep_n = sparse.diags(1.0 - theta_n_to_c, format="csc")
    switch_c_to_n = sparse.diags(theta_c_to_n, format="csc")
    switch_n_to_c = sparse.diags(theta_n_to_c, format="csc")
    return sparse.bmat(
        [
            [kc @ keep_c, switch_n_to_c],
            [switch_c_to_n, kn @ keep_n],
        ],
        format="csc",
    )


def max_ratio(mat: sparse.spmatrix, x: np.ndarray) -> float:
    y = mat @ x
    return float(np.max(y / np.maximum(x, EPS)))


def top_activity(station_ids, names, commute, noncommute, limit=10):
    total = commute + noncommute
    order = np.argsort(-total)[:limit]
    rows = []
    for i in order:
        rows.append(
            {
                "stationID": int(station_ids[i]),
                "name": names[i],
                "commute": float(commute[i]),
                "noncommute": float(noncommute[i]),
                "total": float(total[i]),
                "commute_share": float(commute[i] / total[i]) if total[i] else 0.0,
            }
        )
    return rows


def top_vector(station_ids, names, vec: np.ndarray, limit=10):
    order = np.argsort(-vec)[:limit]
    return [
        {
            "stationID": int(station_ids[i]),
            "name": names[i],
            "score": float(vec[i]),
        }
        for i in order
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metroflow-dir", type=Path, default=Path("../../MetroFlow/MetroFlow"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_np"))
    parser.add_argument("--window", default="all")
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--contact-capacity", type=float, default=1.0)
    parser.add_argument("--eta-steps", type=int, default=101)
    parser.add_argument("--elevation-tol", type=float, default=1e-8)
    parser.add_argument("--cw-smoothing", type=float, default=0.0)
    parser.add_argument("--coupling-mode", choices=["uniform", "degree", "inverse-degree", "activity", "target-activity", "data-share-budget", "data-share", "data-share-feedback", "data-share-balanced-feedback", "data-share-relief"], default="uniform")
    parser.add_argument("--theta-cap", type=float, default=1.0)
    parser.add_argument("--feedback-ratio", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    station_ids, names, neighbors, edge_count = read_station_info(args.metroflow_dir / "stationInfo.csv")
    id_to_idx = {sid: i for i, sid in enumerate(station_ids)}
    windows = parse_windows(args.window)
    commute, noncommute, agg = aggregate_inout(args.metroflow_dir / "metroData_InOutFlow.csv", id_to_idx, windows)

    total = commute + noncommute
    p_c = np.divide(commute, total, out=np.zeros_like(commute), where=total > 0)
    p_n = np.divide(noncommute, total, out=np.zeros_like(noncommute), where=total > 0)
    kc = build_kernel(neighbors, commute, p_c)
    kn = build_kernel(neighbors, noncommute, p_n)
    degree = np.asarray([len(ns) for ns in neighbors], dtype=float)
    rho_c, x_c = spectral_radius(kc)
    rho_n, x_n = spectral_radius(kn)
    rho0 = max(rho_c, rho_n)
    high_layer = "commuting" if rho_c >= rho_n else "noncommuting"
    B0 = build_B(kc, kn, np.zeros_like(p_c), np.zeros_like(p_n))
    baseline_lambda_mmca = args.mu / (args.contact_capacity * rho0)
    baseline_lambda = lambda_c_dmp(B0, args.mu, args.contact_capacity)

    scan_path = args.output_dir / "threshold_scan.csv"
    with scan_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "eta",
            "theta_C_to_N_mean",
            "theta_N_to_C_mean",
            "theta_C_to_N_max",
            "theta_N_to_C_max",
            "rho_B",
            "lambda_c",
            "lambda_c_dmp",
            "lambda_c_mmca",
            "dmp_gain_pct",
            "rho_ratio",
            "threshold_gain_pct",
            "threshold_elevated",
            "cw_a_x",
            "cw_d_y",
            "cw_b_xy",
            "cw_c_yx",
            "cw_margin_left",
            "cw_margin_right",
            "cw_margin",
            "cw_margin_normalized",
            "theorem_4_2_satisfied",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(args.eta_steps):
            eta = step / (args.eta_steps - 1) if args.eta_steps > 1 else 0.0
            if args.coupling_mode in {"data-share", "data-share-feedback", "data-share-balanced-feedback", "data-share-relief"}:
                if args.coupling_mode == "data-share-relief":
                    if high_layer == "commuting":
                        theta_c_to_n = args.omega * eta * p_n
                        theta_n_to_c = np.zeros_like(p_c)
                    else:
                        theta_n_to_c = args.omega * eta * p_c
                        theta_c_to_n = np.zeros_like(p_n)
                elif args.coupling_mode in {"data-share-feedback", "data-share-balanced-feedback"}:
                    feedback = float(np.clip(args.feedback_ratio, 0.0, 1.0))
                    if high_layer == "commuting":
                        theta_c_to_n = args.omega * eta * p_n
                        theta_n_to_c = args.omega * feedback * eta * p_c
                        if args.coupling_mode == "data-share-balanced-feedback":
                            theta_n_to_c = theta_n_to_c * x_c / np.maximum(x_n, EPS)
                    else:
                        theta_n_to_c = args.omega * eta * p_c
                        theta_c_to_n = args.omega * feedback * eta * p_n
                        if args.coupling_mode == "data-share-balanced-feedback":
                            theta_c_to_n = theta_c_to_n * x_n / np.maximum(x_c, EPS)
                elif high_layer == "commuting":
                    theta_c_to_n = args.omega * eta * p_n
                    theta_n_to_c = args.omega * (1.0 - eta) * p_c
                else:
                    theta_n_to_c = args.omega * eta * p_c
                    theta_c_to_n = args.omega * (1.0 - eta) * p_n
                theta_c_to_n = np.minimum(args.theta_cap, theta_c_to_n)
                theta_n_to_c = np.minimum(args.theta_cap, theta_n_to_c)
            else:
                if high_layer == "commuting":
                    mean_c_to_n = args.omega * eta
                    mean_n_to_c = args.omega * (1.0 - eta)
                else:
                    mean_n_to_c = args.omega * eta
                    mean_c_to_n = args.omega * (1.0 - eta)

                if args.coupling_mode == "data-share-budget":
                    weights_c_to_n = p_n
                    weights_n_to_c = p_c
                else:
                    weights_c_to_n = coupling_weights(args.coupling_mode, degree, commute, noncommute)
                    weights_n_to_c = coupling_weights(args.coupling_mode, degree, noncommute, commute)
                theta_c_to_n = allocate_theta(mean_c_to_n, weights_c_to_n, cap=args.theta_cap)
                theta_n_to_c = allocate_theta(mean_n_to_c, weights_n_to_c, cap=args.theta_cap)

            B = build_B(kc, kn, theta_c_to_n, theta_n_to_c)
            rho_b, _ = spectral_radius(B)
            lambda_c_mmca = args.mu / (args.contact_capacity * rho_b)
            lambda_c_dmp_value = lambda_c_dmp(B, args.mu, args.contact_capacity)
            lambda_c = lambda_c_dmp_value

            B_cc = kc @ sparse.diags(1.0 - theta_c_to_n, format="csc")
            B_nn = kn @ sparse.diags(1.0 - theta_n_to_c, format="csc")
            rho_cc, x_diag = spectral_radius(B_cc)
            rho_nn, y_diag = spectral_radius(B_nn)
            smooth = float(np.clip(args.cw_smoothing, 0.0, 1.0))
            uniform = np.full_like(x_diag, 1.0 / len(x_diag))
            x_w = (1.0 - smooth) * x_diag + smooth * uniform
            y_w = (1.0 - smooth) * y_diag + smooth * uniform
            x_w = np.maximum(x_w, EPS)
            y_w = np.maximum(y_w, EPS)
            x_w = x_w / x_w.sum()
            y_w = y_w / y_w.sum()

            if args.coupling_mode == "data-share-relief":
                a_x = rho_cc
                d_y = rho_nn
            else:
                a_x = max_ratio(B_cc, x_w)
                d_y = max_ratio(B_nn, y_w)
            b_xy = float(np.max(theta_n_to_c * y_w / np.maximum(x_w, EPS)))
            c_yx = float(np.max(theta_c_to_n * x_w / np.maximum(y_w, EPS)))
            left = (rho0 - a_x) * (rho0 - d_y)
            right = b_xy * c_yx
            cw_margin = left - right
            cw_scale = abs(left) + abs(right) + EPS
            theorem_ok = (a_x < rho0) and (d_y < rho0) and (left > right)

            writer.writerow(
                {
                    "eta": eta,
                    "theta_C_to_N_mean": float(theta_c_to_n.mean()),
                    "theta_N_to_C_mean": float(theta_n_to_c.mean()),
                    "theta_C_to_N_max": float(theta_c_to_n.max()),
                    "theta_N_to_C_max": float(theta_n_to_c.max()),
                    "rho_B": rho_b,
                    "lambda_c": lambda_c,
                    "lambda_c_dmp": lambda_c_dmp_value,
                    "lambda_c_mmca": lambda_c_mmca,
                    "dmp_gain_pct": (lambda_c_dmp_value / baseline_lambda - 1.0) * 100.0,
                    "rho_ratio": rho_b / rho0,
                    "threshold_gain_pct": (lambda_c_dmp_value / baseline_lambda - 1.0) * 100.0,
                    "threshold_elevated": int(rho_b < rho0 * (1.0 - args.elevation_tol)),
                    "cw_a_x": a_x,
                    "cw_d_y": d_y,
                    "cw_b_xy": b_xy,
                    "cw_c_yx": c_yx,
                    "cw_margin_left": left,
                    "cw_margin_right": right,
                    "cw_margin": cw_margin,
                    "cw_margin_normalized": cw_margin / cw_scale,
                    "theorem_4_2_satisfied": int(theorem_ok),
                }
            )

    summary = {
        "window": args.window,
        "omega": args.omega,
        "coupling_mode": args.coupling_mode,
        "theta_cap": args.theta_cap,
        "feedback_ratio": args.feedback_ratio,
        "cw_smoothing": args.cw_smoothing,
        "mu": args.mu,
        "contact_capacity": args.contact_capacity,
        "station_count": len(station_ids),
        "physical_edge_count": edge_count,
        "avg_physical_degree": 2.0 * edge_count / len(station_ids),
        "aggregation": agg,
        "rho_commuting": rho_c,
        "rho_noncommuting": rho_n,
        "commuting_kernel_nnz": int(kc.nnz),
        "noncommuting_kernel_nnz": int(kn.nnz),
        "mean_commute_participation": float(p_c.mean()),
        "mean_noncommute_participation": float(p_n.mean()),
        "baseline_rho": rho0,
        "baseline_lambda": baseline_lambda,
        "baseline_lambda_dmp": baseline_lambda,
        "baseline_lambda_mmca": baseline_lambda_mmca,
        "high_risk_layer": high_layer,
        "top_activity_stations": top_activity(station_ids, names, commute, noncommute),
        "top_commuting_perron_stations": top_vector(station_ids, names, x_c),
        "top_noncommuting_perron_stations": top_vector(station_ids, names, x_n),
        "outputs": {"threshold_scan": str(scan_path)},
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()














