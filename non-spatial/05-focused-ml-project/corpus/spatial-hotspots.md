# Spatial Hotspots — autocorrelation and cluster detection

Spatial Hotspots tests whether values cluster in space and maps where. It
implements the standard spatial-statistics toolkit in a pure-numpy reference
layer: global Moran's I and Geary's C for overall autocorrelation, Local Moran
(LISA) for per-location clusters and outliers, and Getis-Ord Gi* for hot and
cold spots, each with significance testing. An optional geographically weighted
regression (GWR) path models how relationships vary across space.

The demonstration looks at USDA county-level crop-yield patterns, where the
Moran's I statistic confirms positive spatial autocorrelation and the Gi* map
highlights yield hotspots and coldspots.
