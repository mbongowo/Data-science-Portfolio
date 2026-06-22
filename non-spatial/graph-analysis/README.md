# graph-analysis

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Large-scale graph analytics**: run PageRank (plain, weighted, and
personalized), connected components / label propagation, triangle counting,
betweenness centrality, k-core decomposition, modularity and degree statistics
on a real network, and read the results honestly. The interpretation-critical
algorithms are a pure-numpy reference layer with hand-derived tests; the core
algorithms also run at scale on Spark GraphFrames. Assumptions and what the
numbers do *not* mean are written out, not hand-waved.

---

## Result first

**Question.** On the **LiveJournal social network** (Stanford SNAP,
`soc-LiveJournal1`, ~4.8M nodes / ~69M directed edges), which accounts are most
central by PageRank, how many communities does the graph break into, and how
clustered is it?

**Answer (reproducible demo).** The numbers below are **real**, produced by
`pixi run demo` (equivalently `make demo` or `bdgraph demo`). The demo drives the
exact same pure-numpy core on a small **seeded synthetic stochastic block model**
(SBM) with three planted communities, so the output is verifiable by eye and
deterministic in seconds. The same four algorithms run unchanged at SNAP scale on
Spark GraphFrames over a real network such as LiveJournal.

```
Graph (seeded SBM)  : 30 nodes, 83 edges, 3 planted communities
PageRank (d=0.85)   : top nodes  [ 16, 11, 6, 4, 10 ]  (scores 0.0475, 0.0459, 0.0411, 0.0410, 0.0405)
Components          : 1 connected component (the inter-block edges tie it together)
Label propagation   : 3 communities found  ==  3 planted  (exact recovery)
Triangle counting   : 69 global triangles   avg local clustering ~ 0.505
Betweenness (Brandes): top node 16 (the same hub PageRank flags)
k-core              : max core number 5   degree mean 5.53 / max 9
Modularity (planted): Q = 0.559  (single-community partition gives Q = 0)
```

*Interpretation:* on a clean planted graph the core behaves as it should —
label propagation recovers exactly the 3 planted communities, the planted
partition has high positive modularity (~0.56), the sparse inter-community edges
leave a single connected component, and the dense within-community wiring shows
up as a high local clustering coefficient and a deep k-core. The highest-PageRank
node is also the highest-betweenness node: the best-connected hub of the densest
block.

**Reproduce:** `pixi run demo`  (writes `outputs/pagerank_top.csv`,
`outputs/communities.csv`, `outputs/summary.json`; pinned in `tests/test_demo.py`).

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
  centrality.py           # weighted + personalized PageRank, Brandes betweenness
  components.py           # union-find connected components + num_components
  community.py            # deterministic label propagation (seeded, lowest-label tie)
  structure.py            # k-core decomposition, modularity, degree statistics
  triangles.py            # trace(A^3)/6 global, diag(A^3)/2 per node
  graphframes_pipeline.py # the core algorithms at scale on Spark (lazy imports)
  viz.py                  # sample + draw a small subgraph (lazy networkx/matplotlib)
  demo.py                 # seeded SBM end-to-end demo over the real core (no data)
  cli.py                  # `bdgraph` entry point: demo / pagerank / communities / triangles
notebooks/
  01_walkthrough.ipynb    # runs run_demo(0) and shows betweenness / k-core / modularity
```

The numeric core is pure numpy / stdlib with no third-party dependency, so it is
always importable and is the basis of the tests. It now covers PageRank (plain,
`weighted_pagerank`, and restart-biased `personalized_pagerank`),
`connected_components`, `label_propagation` scored by `modularity`,
`triangle_count`, `betweenness_centrality` (Brandes), `k_core_decomposition` and
`degree_stats`. Each is covered by **hand-derived known-answer tests** whose
expected values are computed by hand on tiny graphs: a symmetric two-node graph
gives PageRank *[1/2, 1/2]* and a directed 3-cycle the uniform *[1/3, 1/3, 1/3]*;
a single triangle counts *1* and *K4* counts *4*; two disjoint cliques resolve to
exactly two communities; an *m*-clique has every k-core number *m-1* and a path
*1*; a star centre has betweenness *(n-1)(n-2)/2*, path endpoints *0*; two
disjoint triangles partitioned as themselves give modularity *Q = 0.5*. Spark
GraphFrames and networkx are imported **lazily** inside their wrappers and are
never touched by the core or the test suite, so the tests run with only numpy,
pandas and pyyaml installed.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: load an edge list, run
the algorithms (at scale for the core four), interpret the central nodes,
communities, cores and betweenness, sample a subgraph for a figure, and a section
on what these statistics do not prove. A short
[`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb) runs the demo
and exercises betweenness, k-core and modularity on the SBM graph.

---

## Run it

### Option A — pixi (recommended; conda-forge resolves the Spark stack)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
pixi run demo           # seeded SBM demo; needs no downloaded data
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
