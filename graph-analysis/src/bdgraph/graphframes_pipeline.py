"""Spark GraphFrames pipeline for graph analytics at scale.

This is the **scale wrapper**. The pure-numpy core in :mod:`bdgraph` runs the
same algorithms on a dense adjacency and is fine up to a few thousand nodes; a
SNAP graph like ``soc-LiveJournal1`` (~69M edges) does not fit a dense matrix,
so it runs here on Spark with the GraphFrames package.

Every heavy import (``pyspark``, ``graphframes``) happens **inside** the
functions, so importing this module costs nothing and the test suite never pulls
in Spark. This module is deliberately not imported by :mod:`bdgraph` or by the
tests.

Run a Spark job with the GraphFrames JAR on the classpath, e.g.::

    pyspark --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12

then call :func:`run` from the driver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame, SparkSession


def load_edges(
    spark: SparkSession,
    edge_file: str,
    *,
    comment: str = "#",
    delimiter: str | None = None,
) -> DataFrame:
    """Read a SNAP edge list into a Spark DataFrame of ``(src, dst)``."""
    from pyspark.sql import functions as fn

    sep = delimiter if delimiter is not None else r"\s+"
    raw = spark.read.text(edge_file)
    raw = raw.filter(~fn.col("value").startswith(comment))
    cols = fn.split(fn.trim(fn.col("value")), sep)
    return raw.select(
        cols.getItem(0).cast("long").alias("src"),
        cols.getItem(1).cast("long").alias("dst"),
    ).na.drop()


def build_graph(edges: DataFrame, *, directed: bool = True) -> Any:
    """Build a GraphFrame from an edge DataFrame.

    For the undirected algorithms (components, triangles) the caller passes the
    edges as stored; GraphFrames symmetrises internally where required. When
    ``directed`` is False the reverse edges are added explicitly so PageRank and
    label propagation see an undirected graph.
    """
    from graphframes import GraphFrame
    from pyspark.sql import functions as fn

    if not directed:
        rev = edges.select(fn.col("dst").alias("src"), fn.col("src").alias("dst"))
        edges = edges.unionByName(rev).dropDuplicates(["src", "dst"])

    vertices = (
        edges.select(fn.col("src").alias("id"))
        .union(edges.select(fn.col("dst").alias("id")))
        .distinct()
    )
    return GraphFrame(vertices, edges)


def run(
    edge_file: str,
    *,
    directed: bool = True,
    damping: float = 0.85,
    max_iter: int = 100,
    comment: str = "#",
    delimiter: str | None = None,
    app_name: str = "bdgraph",
) -> dict[str, Any]:
    """Run PageRank, connected components, and label propagation at scale.

    Returns a dict of Spark DataFrames keyed by algorithm. The caller is
    responsible for collecting, writing, or further aggregating them, and for
    stopping the Spark session.
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName(app_name).getOrCreate()
    edges = load_edges(spark, edge_file, comment=comment, delimiter=delimiter)
    g = build_graph(edges, directed=directed)

    reset = 1.0 - damping
    pr = g.pageRank(resetProbability=reset, maxIter=max_iter)
    cc = g.connectedComponents()
    lpa = g.labelPropagation(maxIter=max_iter)
    tri = g.triangleCount()

    return {
        "pagerank": pr.vertices,
        "connected_components": cc,
        "label_propagation": lpa,
        "triangle_count": tri,
        "_spark": spark,
    }
