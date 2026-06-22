"""bdgraph: large-scale graph analytics with a pure-numpy core.

This package implements its graph algorithms twice. The interpretation-critical
numeric core is a **pure-numpy / stdlib reference layer** with no third-party
dependency, so it is always importable and testable:

* PageRank by power iteration (:func:`pagerank`), plus weight-explicit and
  restart-biased variants (:func:`weighted_pagerank`,
  :func:`personalized_pagerank`),
* connected components by union-find (:func:`connected_components`,
  :func:`num_components`),
* community detection by deterministic label propagation
  (:func:`label_propagation`), scored with :func:`modularity`,
* triangle counting (:func:`triangle_count`, :func:`per_node_triangles`),
* shortest-path betweenness by Brandes' algorithm
  (:func:`betweenness_centrality`),
* k-core decomposition (:func:`k_core_decomposition`) and degree statistics
  (:func:`degree_stats`).

The same algorithms run at scale on Spark GraphFrames in
:mod:`bdgraph.graphframes_pipeline` (lazy Spark imports), and a small subgraph
can be drawn with :mod:`bdgraph.viz` (lazy networkx/matplotlib imports). Neither
of those is imported here, so the core stays dependency-free.
"""

from __future__ import annotations

from bdgraph.centrality import (
    betweenness_centrality,
    personalized_pagerank,
    weighted_pagerank,
)
from bdgraph.community import label_propagation
from bdgraph.components import connected_components, num_components
from bdgraph.pagerank import pagerank
from bdgraph.structure import degree_stats, k_core_decomposition, modularity
from bdgraph.triangles import per_node_triangles, triangle_count

__all__ = [
    "pagerank",
    "weighted_pagerank",
    "personalized_pagerank",
    "connected_components",
    "num_components",
    "label_propagation",
    "modularity",
    "triangle_count",
    "per_node_triangles",
    "betweenness_centrality",
    "k_core_decomposition",
    "degree_stats",
    "__version__",
]

__version__ = "0.1.0"
