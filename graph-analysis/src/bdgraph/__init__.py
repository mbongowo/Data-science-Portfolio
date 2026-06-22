"""bdgraph: large-scale graph analytics with a pure-numpy core.

This package implements four graph algorithms twice. The interpretation-critical
numeric core is a **pure-numpy / stdlib reference layer** with no third-party
dependency, so it is always importable and testable:

* PageRank by power iteration (:func:`pagerank`),
* connected components by union-find (:func:`connected_components`,
  :func:`num_components`),
* community detection by deterministic label propagation
  (:func:`label_propagation`),
* triangle counting (:func:`triangle_count`, :func:`per_node_triangles`).

The same algorithms run at scale on Spark GraphFrames in
:mod:`bdgraph.graphframes_pipeline` (lazy Spark imports), and a small subgraph
can be drawn with :mod:`bdgraph.viz` (lazy networkx/matplotlib imports). Neither
of those is imported here, so the core stays dependency-free.
"""

from __future__ import annotations

from bdgraph.community import label_propagation
from bdgraph.components import connected_components, num_components
from bdgraph.pagerank import pagerank
from bdgraph.triangles import per_node_triangles, triangle_count

__all__ = [
    "pagerank",
    "connected_components",
    "num_components",
    "label_propagation",
    "triangle_count",
    "per_node_triangles",
    "__version__",
]

__version__ = "0.1.0"
