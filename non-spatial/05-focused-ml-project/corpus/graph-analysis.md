# Graph Analysis — large-scale graph algorithms

Graph Analysis implements classic large-scale graph algorithms and runs them on
both a pure-numpy reference and Spark GraphFrames. The algorithms include
PageRank for node importance, label propagation for community detection,
triangle counting, betweenness centrality, k-core decomposition and modularity
scoring. The pure-numpy versions make each algorithm transparent and testable.

The demonstration uses a seeded stochastic block model with planted communities,
where label propagation and modularity recover the planted structure. At scale
the same algorithms run on the SNAP LiveJournal graph of roughly 69 million
edges.
