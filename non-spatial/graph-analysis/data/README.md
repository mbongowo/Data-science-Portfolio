# Data

This project runs on plain edge lists. The reference datasets come from the
**Stanford Network Analysis Project (SNAP)** collection, which is free to
download and widely used as a benchmark.

- SNAP datasets index: https://snap.stanford.edu/data/

## Recommended graphs

| Dataset | Nodes | Edges | Notes |
|---|---|---|---|
| `soc-LiveJournal1` | ~4.8M | ~69M | Directed social network; the scale target |
| `sx-stackoverflow` | ~2.6M | ~63M | Temporal interaction network (src dst t) |
| `ego-Facebook` | ~4K | ~88K | Small, good for a first end-to-end pass |

## Download an edge list

LiveJournal social network (directed, ~69M edges):

```bash
mkdir -p data/raw
curl -L https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz \
  -o data/raw/soc-LiveJournal1.txt.gz
gunzip data/raw/soc-LiveJournal1.txt.gz
# -> data/raw/soc-LiveJournal1.txt
```

The Stack Overflow temporal network (each line is `src dst timestamp`; the
loader reads the first two columns):

```bash
curl -L https://snap.stanford.edu/data/sx-stackoverflow.txt.gz \
  -o data/raw/sx-stackoverflow.txt.gz
gunzip data/raw/sx-stackoverflow.txt.gz
```

## Format

Each non-comment line is one edge, `src dst`, separated by whitespace (or set
`graph.delimiter` in `config/graph.yaml`). Lines starting with `#` are comments
and are skipped. Node ids are integers; they need not be contiguous.

Point `graph.edge_file` in `config/graph.yaml` at the file you downloaded. Raw
data is git-ignored and reproducible from the commands above.
