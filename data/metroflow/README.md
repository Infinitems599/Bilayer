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

The five windows are:

1. 06:00-07:00
2. 07:00-09:00
3. 09:00-17:00
4. 17:00-19:00
5. 19:00-23:00

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
