from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np

from metroflow_dmp import WINDOWS, read_and_audit, read_calendar


HERE = Path(__file__).resolve().parent


def read_nodes(path: Path) -> tuple[list[dict], list[tuple[int, int]]]:
    nodes: list[dict] = []
    station_to_node: dict[int, int] = {}
    raw_neighbors: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for node_id, row in enumerate(reader):
            station_id = int(row["stationID"])
            station_to_node[station_id] = node_id
            raw_neighbors[station_id] = [
                int(value) for value in ast.literal_eval(row["neighbour"])
            ]
            nodes.append(
                {
                    "node_id": node_id,
                    "station_id": station_id,
                    "station_name": row["name"].strip(),
                    "longitude": float(row["lon"]),
                    "latitude": float(row["lat"]),
                }
            )

    edges: set[tuple[int, int]] = set()
    for station_id, neighbors in raw_neighbors.items():
        source = station_to_node[station_id]
        for neighbor_station in neighbors:
            if neighbor_station not in station_to_node:
                continue
            target = station_to_node[neighbor_station]
            if source != target:
                edges.add((min(source, target), max(source, target)))
    return nodes, sorted(edges)


def write_nodes(
    path: Path,
    window: str,
    nodes: list[dict],
    commuting: np.ndarray,
    noncommuting: np.ndarray,
) -> None:
    total = commuting + noncommuting
    commuting_share = np.divide(
        commuting,
        total,
        out=np.zeros_like(commuting),
        where=total > 0,
    )
    noncommuting_share = np.divide(
        noncommuting,
        total,
        out=np.zeros_like(noncommuting),
        where=total > 0,
    )
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# MetroFlow bilayer station network\n")
        stream.write(f"# Time window: {window.replace('--', '-')}\n")
        stream.write(f"# Nodes: {len(nodes)}\n")
        stream.write(
            "# NodeId\tStationId\tStationName\tLongitude\tLatitude\t"
            "CommutingActivity\tNoncommutingActivity\t"
            "CommutingShare\tNoncommutingShare\n"
        )
        for node, c_value, n_value, c_share, n_share in zip(
            nodes,
            commuting,
            noncommuting,
            commuting_share,
            noncommuting_share,
        ):
            stream.write(
                f"{node['node_id']}\t{node['station_id']}\t"
                f"{node['station_name']}\t{node['longitude']:.12g}\t"
                f"{node['latitude']:.12g}\t"
                f"{c_value:.12g}\t{n_value:.12g}\t"
                f"{c_share:.12g}\t{n_share:.12g}\n"
            )


def write_edges(
    path: Path,
    window: str,
    node_count: int,
    edges: list[tuple[int, int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "# Undirected graph (each unordered node pair is saved once): "
            "MetroFlow station adjacency network\n"
        )
        stream.write(f"# Time window: {window.replace('--', '-')}\n")
        stream.write(f"# Nodes: {node_count} Edges: {len(edges)}\n")
        stream.write("# FromNodeId\tToNodeId\n")
        for source, target in edges:
            stream.write(f"{source}\t{target}\n")


def is_connected(
    node_count: int,
    edges: list[tuple[int, int]],
) -> bool:
    adjacent = [[] for _ in range(node_count)]
    for source, target in edges:
        adjacent[source].append(target)
        adjacent[target].append(source)
    visited = {0}
    stack = [0]
    while stack:
        source = stack.pop()
        for target in adjacent[source]:
            if target not in visited:
                visited.add(target)
                stack.append(target)
    return len(visited) == node_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the five processed MetroFlow networks in SNAP-style text files."
    )
    parser.add_argument("--metroflow-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "data" / "metroflow",
    )
    args = parser.parse_args()

    nodes, edges = read_nodes(args.metroflow_dir / "stationInfo.csv")
    station_ids = [int(node["station_id"]) for node in nodes]
    calendar = read_calendar(
        args.metroflow_dir / "MetaData" / "workday_calendar.csv"
    )
    commuting, noncommuting, audit = read_and_audit(
        args.metroflow_dir,
        station_ids,
        calendar,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for window_index, (window, _, _) in enumerate(WINDOWS):
        tag = (
            f"{window_index + 1:02d}_"
            + window.replace(":", "").replace("--", "_")
        )
        node_path = args.output_dir / f"metroflow_{tag}_nodes.txt"
        edge_path = args.output_dir / f"metroflow_{tag}_edges.txt"
        write_nodes(
            node_path,
            window,
            nodes,
            commuting[0, window_index],
            noncommuting[0, window_index],
        )
        write_edges(edge_path, window, len(nodes), edges)
        files.append(
            {
                "window": window,
                "nodes": node_path.name,
                "edges": edge_path.name,
                "commuting_activity": float(
                    commuting[0, window_index].sum()
                ),
                "noncommuting_activity": float(
                    noncommuting[0, window_index].sum()
                ),
            }
        )

    metadata = {
        "format": "SNAP-style tab-separated text",
        "directed": False,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "connected": is_connected(len(nodes), edges),
        "networks": files,
        "source_rows": int(audit["actual_records"]),
        "duplicate_rows": int(audit["duplicate_station_time_records"]),
        "unexpected_keys": int(
            audit["unexpected_date_slot_station_keys"]
        ),
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"metadata": str(metadata_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
