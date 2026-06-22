# graph-analysis

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Large-scale graph analytics**: run PageRank, connected components / label
propagation, and triangle counting on a real network, and read the results
honestly. The interpretation-critical algorithms are a pure-numpy reference
layer with hand-derived tests; the same algorithms run at scale on Spark
GraphFrames. Assumptions and what the numbers do *not* mean are written out, not
hand-waved.

---

## Result first

**Question.** On the **LiveJournal social network** (Stanford SNAP,
`soc-LiveJournal1`, ~4.8M nodes / ~69M directed edges), which accounts are most
central by PageRank, how many communities does the graph break into, and how
clustered is it?

**Answer (illustrative).** A handful of accounts dominate the PageRank mass; the
graph is essentially one giant weakly-connected component with a long tail of
tiny ones; and the global triangle count is high relative to a random graph of
the same density, so the network is strongly locally clustered — friends of
friends are friends.

![Placeholder subgraph sample](outputs/.gitkeep)
<!-- Running the Spark pipeline writes outputs/*.parquet; sample a subgraph with
     bdgraph.viz and drop the PNG here. -->

```
PageRank (d=0.85)   : top nodes  [ 12, 4815, 991, 30172, 7 ]  (mass concentrated)
Components          : 1 giant weakly-connected component covers > 99% of nodes
Label propagation   : ~ thousands of communities, heavy-tailed in size
Triangle counting   : global triangles ~ 2.85e8   avg clustering ~ 0.27
```

*(Numbers above are illustrative placeholders; run the pipeline to regenerate
them for the graph you downloaded.)*

### What this analysis does **not** let you conclude

- **Centrality is not importance.** A high PageRank node is visited often by one
  specific random walk. It is not "the most important account" in any absolute
  sense; change the damping factor or the teleport set and the ranking changes.
- **Communities have a resolution limit.** Label propagation (and modularity
  methods generally) merge small genuine communities and can split large ones.
  The detected partition is one heuristic answer, not the partition.
- **The picture is a sample.** A 69M-edge graph cannot be drawn; any figure is a
  sampled subgraph, and breadth-first or ego samples over-represent hubs. Do not
  read structure off the picture that the statistics do not support.
- **Direction matters and is treated differently per algorithm.** PageRank
  respects edge direction; connected components and triangle counting treat the
  graph as undirected. A pair in the same weak component need not reach each
  other respecting direction.
- **Counts describe structure, not cause.** Triangles measure local link
  density; they say nothing about why edges formed.

---

## How it works

```
data/README.md            # how to fetch a SNAP edge list (soc-LiveJournal1, ...)
        |
config/graph.yaml         # edge file, directed flag, damping, max_iter, tol, algos
        |
src/bdgraph/
  pagerank.py             # power iteration on a dense adjacency, dangling-node safe
  components.py           # union-find connected components + num_components
  community.py            # deterministic label propagation (seeded, lowest-label tie)
  triangles.py            # trace(A^3)/6 global, diag(A^3)/2 per node
  graphframes_pipeline.py # the same algorithms at scale on Spark (lazy imports)
  viz.py                  # sample + draw a small subgraph (lazy networkx/matplotlib)
  cli.py                  # `bdgraph` entry point: pagerank / communities / triangles
```

The numeric core (`pagerank`, `connected_components`, `label_propagation`,
`triangle_count`) is pure numpy / stdlib with no third-party dependency, so it
is always importable and is the basis of the tests. It is covered by
**hand-derived known-answer tests** whose expected values are computed by hand
on tiny graphs: a symmetric two-node graph gives PageRank *[1/2, 1/2]*, a
directed 3-cycle gives the uniform *[1/3, 1/3, 1/3]*; a single triangle counts
*1* and the complete graph *K4* counts *4*; two disjoint cliques resolve to
exactly two communities. Spark GraphFrames and networkx are imported **lazily**
inside their wrappers and are never touched by the core or the test suite, so
the tests run with only numpy, pandas and pyyaml installed.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: load an edge list, run
the four algorithms at scale, interpret the central nodes and communities,
sample a subgraph for a figure, and a section on what these statistics do not
prove.

---

## Run it

### Option A — pixi (recommended; conda-forge resolves the Spark stack)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
# download an edge list first (see data/README.md), then:
pixi run pagerank
pixi run communities
pixi run triangles
```

> Note: `pixi.lock` is **machine-generated**. It is not committed here; running
> `pixi install` creates it on your platform.

### Option B — pip / venv

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
make pagerank
make communities
make triangles
```

### Option C — Docker

```bash
docker build -t graph-analysis .
docker run --rm graph-analysis        # runs the test suite
```

GraphFrames at scale runs on a Spark cluster separately; the image is for the
pure-numpy core and the tests, not for distributed jobs.

---

## Configuration

Everything analysis-defining lives in [`config/graph.yaml`](config/graph.yaml):
the edge file path, the directed flag, the PageRank damping / `max_iter` / `tol`,
the community method and seed, and the list of algorithms to run.

---

## Data sources

- **Stanford Network Analysis Project (SNAP)** — free graph datasets. The scale
  target is **LiveJournal** (`soc-LiveJournal1`, ~69M edges); the **Stack
  Overflow temporal network** (`sx-stackoverflow`) is an alternative. See
  [`data/README.md`](data/README.md) for download commands.

Raw data and outputs are git-ignored and regenerated by the download commands
and the pipeline.

---

## License

MIT © 2026 Joseph Mbuh
