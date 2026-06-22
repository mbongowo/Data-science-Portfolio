# Usage guide: the graph-analytics workflow

This guide walks through one pass of graph analysis with this repository: install
the stack, get an edge list, build the graph, run PageRank, connected components,
label propagation and triangle counting at scale, interpret the central nodes and
communities, sample a representative subgraph for a figure, and close with what
these statistics do not establish.

The pure-numpy reference functions (`pagerank`, `weighted_pagerank`,
`personalized_pagerank`, `connected_components`, `label_propagation`,
`modularity`, `triangle_count`, `per_node_triangles`, `betweenness_centrality`,
`k_core_decomposition`, `degree_stats`) run with only numpy installed and operate
on a **dense** adjacency, so they are for small problems (up to a few thousand
nodes) and for checking your understanding. A SNAP graph with tens of millions of
edges does not fit a dense matrix; the four core algorithms (PageRank, connected
components, label propagation, triangle counting) run on Spark in
`bdgraph.graphframes_pipeline`, described below.

## 1. Install

The Spark stack (pyspark plus a matching GraphFrames JAR, and a JVM) resolves
most reliably through conda-forge. Pixi is the path the repository is set up for.

```bash
pixi install        # resolves dependencies and writes pixi.lock locally
pixi run test       # confirm the install: the test suite should pass
```

If you prefer pip, expect to provide a JVM yourself and to match the GraphFrames
JAR to your Spark version, then:

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

A quick check that the numeric core is importable without Spark or networkx:

```bash
python -c "import numpy; from bdgraph import pagerank; print('ok')"
```

## 2. Get an edge list

The reference graphs come from the Stanford Network Analysis Project (SNAP).
Download one as described in [`data/README.md`](data/README.md). The scale target
is LiveJournal:

```bash
mkdir -p data/raw
curl -L https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz \
  -o data/raw/soc-LiveJournal1.txt.gz
gunzip data/raw/soc-LiveJournal1.txt.gz
```

Each non-comment line is one edge, `src dst`. Lines starting with `#` are
comments and are skipped. Point `graph.edge_file` in `config/graph.yaml` at the
file you downloaded, and set `graph.directed` to match the dataset (LiveJournal
is directed).

## 3. From edges to a graph

The CLI reads the edge list, re-indexes node ids into a contiguous `0..n-1`
space, and builds the structure each algorithm needs. PageRank uses the directed
adjacency; connected components and triangle counting symmetrise it (an edge
joins its endpoints regardless of stored direction). For a small graph you can
do this directly:

```python
import numpy as np
from bdgraph import pagerank, connected_components, label_propagation, triangle_count

# dense adjacency for a tiny directed graph: 0 -> 1 -> 2 -> 0, plus 2 -> 3
adj = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 1],
    [0, 0, 0, 0],
], dtype=float)

pr = pagerank(adj, damping=0.85)        # probability vector, sums to 1
edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
labels = connected_components(4, edges) # union-find component labels
comm = label_propagation(adj)           # deterministic community labels
t = triangle_count(adj)                 # global triangle count (undirected)
```

The dense path is fine here. For LiveJournal, skip to section 7.

## 4. PageRank

PageRank is the stationary distribution of a random surfer who follows an
out-edge with probability `damping` and teleports uniformly with probability
`1 - damping`. The reference implementation is power iteration on the dense
transition matrix, with dangling nodes (no out-edges) redistributing their mass
uniformly so the vector always sums to 1.

```python
pr = pagerank(adj, damping=0.85, max_iter=100, tol=1e-9)
top = pr.argsort()[::-1][:10]   # indices of the ten highest-ranked nodes
```

How to read it:

- The vector is a probability distribution: non-negative, sums to 1. A node's
  score is its steady-state visit frequency under *this* walk.
- `damping` is part of the model. The classic 0.85 means the surfer teleports
  15% of the time; lower damping flattens the ranking, higher damping
  concentrates it on sink-like regions. Report the value you used.
- Direction matters. Reverse the edges, or treat the graph as undirected, and the
  ranking generally changes.

## 5. Connected components and communities

Two different questions. **Connected components** ask who is reachable from whom
under the undirected edge set; **communities** ask which densely interconnected
groups the graph breaks into.

```python
from bdgraph import num_components

labels = connected_components(n, edges)   # one label per node
k = num_components(n, edges)              # how many components
comm = label_propagation(adj, max_iter=10, seed=0)
```

- Component labels are canonical: each component is named by its smallest member
  id, so the labelling is deterministic. On a directed graph this is *weak*
  connectivity — same component does not imply each can reach the other
  respecting direction.
- Label propagation gives every node its own label, then each round adopts the
  neighbour label carrying the most edge weight; ties break to the **lowest**
  label, and a fixed seed sets the visit order, so the partition is reproducible
  run to run. It is a heuristic with a resolution limit (see section 8); two
  disjoint cliques resolve to exactly two communities, which is the known-answer
  the tests check.

## 6. Triangle counting and clustering

For an undirected simple graph with binary adjacency `A`, the number of closed
triangles is `trace(A^3) / 6` and the number through node `i` is
`diag(A^3)[i] / 2`.

```python
from bdgraph import triangle_count, per_node_triangles

total = triangle_count(adj)          # global count
per_node = per_node_triangles(adj)   # triangles through each node
assert per_node.sum() == 3 * total   # each triangle touches three nodes
```

A high global count relative to a random graph of the same density means the
network is locally clustered. The per-node counts feed the local clustering
coefficient if you want it.

## 6b. Weighted and personalized PageRank

Plain `pagerank` already honours edge weights by normalising each node's
out-edges by its out-strength; `weighted_pagerank` is the same numerics under an
explicit name. `personalized_pagerank` adds a **restart** (personalization)
vector: when the surfer teleports it lands on that distribution rather than
uniformly, so the ranking is biased toward the restart set.

```python
from bdgraph import weighted_pagerank, personalized_pagerank

wp = weighted_pagerank(adj, damping=0.85)          # heavier edges carry more mass
restart = [1, 0, 0, 0]                             # bias toward node 0 (auto-normalised)
ppr = personalized_pagerank(adj, restart, damping=0.85)
```

A uniform restart reproduces plain PageRank exactly. Personalized PageRank
answers "important *relative to this restart set*", not "important in general":
move the restart mass and the ranking moves with it.

## 6c. Betweenness, k-core and modularity

Three more structural diagnostics, all exact on the dense reference path.

```python
from bdgraph import betweenness_centrality, k_core_decomposition, modularity, degree_stats

bc = betweenness_centrality(adj, normalized=True)  # Brandes, undirected, [0, 1]
core = k_core_decomposition(adj)                   # core number per node
q = modularity(adj, labels)                        # partition quality vs. null
stats = degree_stats(adj)                          # mean/max/min degree + histogram
```

- **Betweenness** (Brandes' algorithm) is the share of shortest paths a node sits
  on. The raw form gives a star centre `(n-1)(n-2)/2` and path endpoints `0`; the
  normalised form divides by the pair count so scores lie in `[0, 1]`. It is
  exact but `O(nm)`, so it is for small graphs, not SNAP-scale networks.
- **k-core** peels the lowest-degree node repeatedly; a node's core number is the
  deepest core it survives in. A clique `K_m` has every core number `m-1`, a path
  `1`, a cycle `2`. Deep cores flag densely mutually-connected regions.
- **Modularity** scores a partition against the configuration-model null:
  positive means denser-within than chance. Two disjoint triangles partitioned as
  themselves give `Q = 0.5`; one big community gives `Q = 0`. Modularity has a
  resolution limit, so a higher `Q` is "denser-within", not automatically
  "better".

## 7. Running at scale on Spark GraphFrames

The dense core cannot hold a 69M-edge graph. The Spark pipeline runs the same
algorithms distributed. Launch Spark with the GraphFrames package on the
classpath and call `run` from the driver:

```bash
pyspark --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12
```

```python
from bdgraph import graphframes_pipeline as gp

results = gp.run(
    "data/raw/soc-LiveJournal1.txt",
    directed=True, damping=0.85, max_iter=20,
)
results["pagerank"].orderBy("pagerank", ascending=False).show(10)
results["connected_components"].groupBy("component").count().show()
results["triangle_count"].agg({"count": "sum"}).show()
results["_spark"].stop()
```

Every Spark and GraphFrames import happens inside these functions, so importing
the module costs nothing and the test suite never pulls in Spark.

Runtime and scaling notes: PageRank and label propagation are iterative and
shuffle-heavy; cost grows with edges times iterations, so cap `max_iter` and
watch the shuffle. Triangle counting is the most expensive — it materialises
two-hop paths and is dominated by high-degree hubs, so a few super-nodes can
skew the job. Connected components converges fast on graphs with one giant
component. Cache the edge DataFrame once and reuse it across the three jobs.

## 8. Sampling a subgraph for visualisation

You cannot draw the whole graph. Extract a small connected subgraph and lay it
out for a sanity-check figure, not for analysis:

```python
from bdgraph import viz

g = viz.sample_subgraph(edges, seed_node=12, max_nodes=200, seed=0)
viz.draw(g, out_path="outputs/subgraph.png", node_attr={n: pr[n] for n in g})
```

networkx and matplotlib are imported lazily inside these functions. A
breadth-first or ego sample over-represents hubs and the seed's neighbourhood, so
the picture is illustration, not an unbiased view of the graph.

## 9. How to interpret responsibly

These statistics describe graph structure. They do not explain it, and a few
limits should travel with any result.

**Centrality is not importance.** PageRank measures visit frequency under one
random walk with one damping factor. A high score flags a node that the walk
returns to often, not an account that matters in any external sense. Change the
damping or the teleport set and the ranking moves.

**Communities have a resolution limit.** Label propagation, like modularity
optimisation, can merge small genuine communities and split large ones, and on
near-symmetric graphs it can oscillate. The deterministic tie-break removes the
randomness, not the bias. Treat the partition as one heuristic answer and check
whether the headline groups survive a different method or seed.

**The visualisation is a sample.** Any drawn subgraph is a biased excerpt.
Structure you read off the picture must be backed by the statistics on the full
graph, not the other way round.

**Direction is handled differently per algorithm.** PageRank respects edge
direction; components and triangles treat the graph as undirected. Be explicit
about which question you are asking, because "connected" and "reachable" are not
the same on a directed graph.

**Counts describe, they do not cause.** A high triangle count says the graph is
locally clustered. It does not say why the edges formed; pattern is a prompt for
explanation, not the explanation.
