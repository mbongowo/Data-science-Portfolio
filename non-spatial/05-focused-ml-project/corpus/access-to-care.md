# Access to Care — travel-time accessibility and coverage equity

Access to Care measures how easily people can reach health facilities. It runs a
multi-source Dijkstra shortest-path over a road network to compute travel time
from populated places to the nearest clinic, weights those times by population,
and reports coverage equity. The equity metrics include a Gini coefficient over
access times and two-step floating catchment area (2SFCA) accessibility scores.

The case study is health-facility access in Cameroon. The pipeline runs on both
synthetic networks for reproducible tests and real road-network routing. The
companion Leafmap Dashboard is a lighter, straight-line screening version of the
same access question.
