# Processed MetroFlow networks

This directory contains the five MetroFlow station networks used in the
manuscript. The files follow the SNAP convention: metadata lines start with
`#`, data are tab separated, and the final comment line gives the column
names.

Each time window has two files:

- `*_nodes.txt`: `NodeId`, original `StationId`, station name, longitude, latitude,
  commuting and non-commuting activity, and the two corresponding local
  activity shares.
- `*_edges.txt`: one undirected physical station edge per line as
  `FromNodeId` and `ToNodeId`; each unordered pair is stored once.

All five networks contain 302 nodes and 349 undirected edges. They share the
same physical topology; their node activities differ by time window. Node IDs
are zero-based and are the IDs referenced by the edge files. The activity
values were aggregated from the released, cleaned MetroFlow table without
imputing missing or zero flow records.

## Dataset statistics

The following statistics apply to the physical topology shared by all five
time-window networks.

| Statistic | Value |
| --- | ---: |
| Graph type | Undirected, connected |
| Nodes | 302 |
| Edges | 349 |
| Nodes in largest connected component | 302 (1.000) |
| Edges in largest connected component | 349 (1.000) |
| Average degree | 2.3113 |
| Graph density | 0.007679 |
| Average clustering coefficient | 0.02196 |
| Number of triangles | 9 |
| Diameter | 41 |
| Average shortest-path length | 14.9764 |
| 90-percentile effective diameter | 25 |

## Time-window networks

| Window | Nodes | Edges | Commuting activity | Non-commuting activity | Total activity | Commuting share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 06:00-07:00 | 302 | 349 | 17,277,040 | 25,672,685 | 42,949,725 | 40.23% |
| 07:00-09:00 | 302 | 349 | 162,569,992 | 160,675,940 | 323,245,932 | 50.29% |
| 09:00-17:00 | 302 | 349 | 107,928,314 | 464,267,504 | 572,195,818 | 18.86% |
| 17:00-19:00 | 302 | 349 | 102,555,715 | 171,463,669 | 274,019,384 | 37.43% |
| 19:00-23:00 | 302 | 349 | 76,615,675 | 168,710,684 | 245,326,359 | 31.23% |

`metadata.json` records filenames, counts, and activity totals. The files can
be regenerated from the released processed MetroFlow directory with:

```powershell
uv run python export_metroflow_snap.py `
  --metroflow-dir D:\path\to\MetroFlow `
  --output-dir data\metroflow
```

The text layout follows the
[Stanford SNAP dataset convention](https://snap.stanford.edu/data/email-Enron.html).

Data source: P. Sun, J. Yang, Z. Huang, et al., "Human Mobility Datasets in
the Complex Metro System of Shanghai," *Scientific Data* 12, 1061 (2025),
[doi:10.1038/s41597-025-05416-8](https://doi.org/10.1038/s41597-025-05416-8).
