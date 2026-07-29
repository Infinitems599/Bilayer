"""Calculation-only generator for the first-order Figure 8 grid.

This module deliberately writes numerical tables only.  Figure styling and PDF
generation are excluded from the public calculation package.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs, splu

from dmp_core import EPS, spectral_radius


@dataclass(frozen=True)
class Figure8Config:
    """Parameters of the first-order phase-diagram calculation."""

    node_count: int = 80
    degree_min: float = 4.0
    degree_max: float = 22.0
    grid_size: int = 25
    instances: int = 10
    ratios: tuple[float, ...] = (4.0, 8.0, 12.0)
    network_types: tuple[str, ...] = ("ER", "WS", "BA")
    omega: float = 0.62
    beta_mean_share: float = 1.0
    recovery_probability: float = 0.3
    seed: int = 13000


def _as_csr(matrix: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    return sparse.csr_matrix(matrix, dtype=float)


def _column_kernel(adjacency: sparse.spmatrix, source_share: np.ndarray) -> sparse.csr_matrix:
    adjacency = _as_csr(adjacency)
    degree = np.asarray(adjacency.sum(axis=0)).ravel()
    inverse = np.divide(
        source_share,
        degree,
        out=np.zeros_like(source_share, dtype=float),
        where=degree > EPS,
    )
    return (adjacency @ sparse.diags(inverse, format="csr")).tocsr()


def _activity_profile(adjacency: sparse.spmatrix, mean: float, heterogeneity: float) -> np.ndarray:
    degree = np.asarray(adjacency.sum(axis=0)).ravel().astype(float)
    relative_degree = degree / max(float(degree.mean()), EPS)
    values = np.clip(mean * (1.0 + heterogeneity * (relative_degree - 1.0)), 0.1 * mean, 3.0 * mean)
    return values * mean / max(float(values.mean()), EPS)


def _synthetic_layers(config: Figure8Config, network_type: str, degree_alpha: float, degree_beta: float, seed: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    n = config.node_count
    if network_type == "BA":
        first = nx.barabasi_albert_graph(n, max(1, int(round(degree_alpha / 2))), seed=seed)
        second = nx.barabasi_albert_graph(n, max(1, int(round(degree_beta / 2))), seed=seed + 7919)
    elif network_type == "WS":
        def even_degree(value: float) -> int:
            k = max(2, int(round(value)))
            k += k % 2
            return min(k, n - 1 if (n - 1) % 2 == 0 else n - 2)
        first = nx.connected_watts_strogatz_graph(n, even_degree(degree_alpha), 0.16, tries=80, seed=seed)
        second = nx.connected_watts_strogatz_graph(n, even_degree(degree_beta), 0.16, tries=80, seed=seed + 7919)
    elif network_type == "ER":
        first = nx.erdos_renyi_graph(n, min(degree_alpha / (n - 1), 1.0), seed=seed)
        second = nx.erdos_renyi_graph(n, min(degree_beta / (n - 1), 1.0), seed=seed + 7919)
        # Keep the message-space dimension well-defined for sparse draws.
        for graph in (first, second):
            if not nx.is_connected(graph):
                graph.add_edges_from(nx.cycle_graph(n).edges())
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    return (
        _as_csr(nx.to_scipy_sparse_array(first, format="csr", dtype=float)),
        _as_csr(nx.to_scipy_sparse_array(second, format="csr", dtype=float)),
    )


def _contact_matrix(a_alpha: sparse.spmatrix, a_beta: sparse.spmatrix, ratio: float, config: Figure8Config, alpha_share: np.ndarray, beta_share: np.ndarray) -> sparse.csr_matrix:
    n = config.node_count
    k_alpha = _column_kernel(a_alpha, alpha_share)
    k_beta = _column_kernel(a_beta, beta_share)
    theta_alpha_beta = np.clip(config.omega * ratio / (1.0 + ratio), 0.0, 1.0)
    theta_beta_alpha = np.clip(config.omega / (1.0 + ratio), 0.0, 1.0)
    identity = sparse.eye(n, format="csr")
    return sparse.bmat(
        [[k_alpha * (1.0 - theta_alpha_beta), identity * theta_beta_alpha],
         [identity * theta_alpha_beta, k_beta * (1.0 - theta_beta_alpha)]],
        format="csr",
    )


def _certificate(matrix: sparse.spmatrix, layer_size: int, mu: float) -> bool:
    """Evaluate the first-order form of the Theorem 4.2 Schur condition."""
    matrix = matrix.tocsc()
    aa, ab = matrix[:layer_size, :layer_size], matrix[:layer_size, layer_size:]
    ba, bb = matrix[layer_size:, :layer_size], matrix[layer_size:, layer_size:]
    if spectral_radius(aa)[0] >= mu or spectral_radius(bb)[0] >= mu:
        return False
    if ab.nnz == 0 or ba.nnz == 0:
        return True
    solve_aa = splu(mu * sparse.eye(layer_size, format="csc") - aa).solve
    solve_bb = splu(mu * sparse.eye(layer_size, format="csc") - bb).solve

    def multiply(x: np.ndarray) -> np.ndarray:
        return solve_bb(ba @ solve_aa(ab @ x))

    loop = LinearOperator((layer_size, layer_size), matvec=multiply, dtype=float)
    value = eigs(loop, k=1, which="LM", return_eigenvectors=False, v0=np.ones(layer_size))[0]
    return bool(abs(value) < 1.0 - 1e-8)


def _seed(config: Figure8Config, network_type: str, alpha_index: int, beta_index: int, instance: int) -> int:
    return config.seed + {"ER": 0, "WS": 20000, "BA": 40000}[network_type] + instance + 41 * alpha_index + 811 * beta_index


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_figure8_data(output_dir: str | Path, config: Figure8Config = Figure8Config()) -> dict[str, Path]:
    """Compute and export the Figure 8 numerical grid.

    The result contains one row per network realization and a second table
    averaged over realizations.  Both are CSV files suitable for an external
    plotting workflow.  No figure is generated here.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    degrees = np.linspace(config.degree_min, config.degree_max, config.grid_size)
    records: list[dict] = []
    heterogeneity = {"ER": 0.30, "WS": 0.48, "BA": 0.70}
    for network_type in config.network_types:
        for alpha_index, degree_alpha in enumerate(degrees):
            for beta_index, degree_beta in enumerate(degrees):
                for instance in range(config.instances):
                    first, second = _synthetic_layers(config, network_type, float(degree_alpha), float(degree_beta), _seed(config, network_type, alpha_index, beta_index, instance))
                    alpha_share = _activity_profile(first, 1.0, heterogeneity[network_type])
                    beta_share = _activity_profile(second, config.beta_mean_share, heterogeneity[network_type])
                    k_alpha = _column_kernel(first, alpha_share)
                    rho_alpha = spectral_radius(k_alpha)[0]
                    for ratio in config.ratios:
                        coupled = _contact_matrix(first, second, ratio, config, alpha_share, beta_share)
                        rho = spectral_radius(coupled)[0]
                        # The theorem analogue is evaluated at the uncoupled
                        # first-order threshold lambda_0=mu/(C*rho_alpha);
                        # the common factor C is absorbed into this
                        # first-order matrix scaling.
                        # Therefore its block inequalities apply to
                        # H_c(lambda_0)=(mu/rho_alpha) B_c after that
                        # absorption, not to B_c itself.
                        certificate_matrix = coupled * (
                            config.recovery_probability / rho_alpha
                        )
                        records.append({
                            "network_type": network_type,
                            "r": ratio,
                            "degree_alpha": degree_alpha,
                            "degree_beta": degree_beta,
                            "instance": instance,
                            "seed": _seed(config, network_type, alpha_index, beta_index, instance),
                            "rho_ratio": rho / rho_alpha,
                            "threshold_gain_pct": 100.0 * (
                                rho_alpha / rho - 1.0
                            ),
                            "theorem_4_2_satisfied": _certificate(certificate_matrix, config.node_count, config.recovery_probability),
                        })
    instance_path = output / "figure8_instances.csv"
    _write_csv(instance_path, records)
    grouped: dict[tuple, list[dict]] = {}
    for row in records:
        key = tuple(row[name] for name in ("network_type", "r", "degree_alpha", "degree_beta"))
        grouped.setdefault(key, []).append(row)
    grid_rows = []
    for key, group in grouped.items():
        grid_rows.append({
            "network_type": key[0], "r": key[1], "degree_alpha": key[2], "degree_beta": key[3],
            "rho_ratio_mean": float(np.mean([row["rho_ratio"] for row in group])),
            "threshold_gain_mean_pct": float(np.mean([row["threshold_gain_pct"] for row in group])),
            "theorem_4_2_all": bool(all(row["theorem_4_2_satisfied"] for row in group)),
            "instances": len(group),
        })
    grid_path = output / "figure8_grid.csv"
    _write_csv(grid_path, grid_rows)
    settings_path = output / "figure8_settings.json"
    settings_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return {"instances": instance_path, "grid": grid_path, "settings": settings_path}
