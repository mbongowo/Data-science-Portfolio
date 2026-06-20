# Usage guide: the ESDA workflow

This guide walks through one pass of exploratory spatial data analysis with this
repository: install the stack, get the data, build and compare spatial weights,
run a global Moran's I, map local clusters with LISA and Getis-Ord Gi*, and
optionally fit a GWR. It closes with what these statistics do not establish.

The pure-numpy reference functions (`morans_i_dense`, `gearys_c_dense`,
`local_moran_dense`, `lisa_quadrants`, `getis_ord_g_star_dense`) run with only
numpy installed and are meant for small problems and for checking your
understanding. The permutation-based inference that real work needs lives in the
`esda`-backed wrappers, which require the full pysal stack described below.

## 1. Install

The geospatial stack (geopandas, libpysal, esda, splot, mgwr and their compiled
GDAL/PROJ dependencies) resolves most reliably through conda-forge. Pixi is the
path the repository is set up for.

```bash
pixi install        # resolves dependencies and writes pixi.lock locally
pixi run test       # confirm the install: the test suite should pass
```

If you prefer pip, expect to install GDAL/PROJ yourself first, then:

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

A quick check that the numeric core is importable without any geo libraries:

```bash
python -c "import numpy; from hotspots import morans_i_dense; print('ok')"
```

## 2. Get a NASS_API_KEY and download the data

The primary dataset joins USDA NASS county crop yield to TIGER/Line county
polygons. The NASS API needs a free key.

1. Request a key at https://quickstats.nass.usda.gov/api. It arrives by email.
2. Put it in the environment:

   ```bash
   export NASS_API_KEY=your_key_here       # Windows PowerShell: $env:NASS_API_KEY="your_key_here"
   ```

3. Edit `config/aoi.yaml` if you want a different state, year, or commodity.
   The defaults are Iowa corn yield for 2023.
4. Run the loader:

   ```bash
   python data/download.py --config config/aoi.yaml --out data/raw
   ```

   This writes `data/raw/iowa_corn_yield_2023.gpkg` (one layer: county polygons
   plus the `yield_bu_acre` column). If the join returns zero rows, the year or
   state in the config does not match any county-level NASS records; change them
   and rerun.

An alternative Landsat surface-temperature path is documented in
`data/download.py` (`_landsat_alternative_note`). The ESDA steps below are the
same; only the loader differs.

## 3. Build and compare spatial weights

Every statistic that follows is conditional on the weights matrix W, which says
which units are neighbours and how strongly. Treat the choice as part of the
model. The repository builds three families and reports neighbour diagnostics so
you can defend the choice.

| Weights | Neighbour rule | When it fits | Watch for |
|---|---|---|---|
| Queen contiguity | Shares an edge or a corner | Irregular polygons that tile a region (counties, tracts) where adjacency is the natural relation | Topology errors; islands (units with no contiguous neighbour) |
| Distance band | Within a fixed radius | Point-like or evenly spaced units; when "near" has a real distance meaning | Needs a projected CRS in metres; a too-small radius leaves islands, a too-large one connects everything |
| K-nearest neighbours | The `k` closest centroids | Uneven unit sizes, or when you want a guaranteed neighbour count and no islands | Imposes a fixed, asymmetric cardinality that may not match real adjacency |

Build and compare them:

```python
import geopandas as gpd
from hotspots import weights as w

gdf = gpd.read_file("data/raw/iowa_corn_yield_2023.gpkg").to_crs("EPSG:5070")

w_queen = w.queen_contiguity(gdf, row_standardize=True)        # raises on islands
w_dist  = w.distance_band(gdf, threshold=None)                 # auto min threshold
w_knn   = w.knn(gdf, k=8)

for d in w.compare_weights({"queen": w_queen, "dist": w_dist, "knn8": w_knn}):
    print(d.as_dict())
```

Row standardisation (`transform="r"`, the default) makes each row sum to one, so
the spatial lag becomes a local average. That is the usual convention for ESDA
and is what the LISA quadrant logic assumes.

Read the diagnostics before trusting any map. A mean neighbour count near one or
a nonzero island count means the weights are starving the statistic. If Queen
produces islands, either widen to a distance band that connects them, switch to
KNN, or drop the isolated units on purpose and say so.

## 4. Global Moran's I

Global Moran's I is one number summarising whether like values sit near like
values across the whole map.

```python
from hotspots import esda

gm = esda.global_moran(gdf["yield_bu_acre"].to_numpy(float), w_queen,
                       permutations=999, seed=42)
print(gm.I, gm.expected_I, gm.p_sim, gm.z_sim)
```

How to read it:

- `I` is the statistic. Its null expectation is `expected_I = -1/(n-1)`, slightly
  below zero, not zero. Compare `I` against that, not against 0.
- Positive `I` above the expectation means clustering: high values neighbour
  high, low neighbour low. Negative means a checkerboard tendency.
- `p_sim` is a pseudo p-value from `permutations` conditional permutations of the
  values across locations. With 999 permutations the smallest achievable value
  is 0.001. Report the permutation count and the seed; both affect the number.

The Moran scatterplot (`esda.moran_scatterplot`) plots each unit's value against
its spatial lag. The slope of the fitted line is Moran's I. Points in the upper
right (high value, high lag) are HH; lower left are LL; the off-diagonal
quadrants are spatial outliers. A tight positive slope is visible clustering; a
formless cloud is near independence.

## 5. Local clusters: LISA and Getis-Ord Gi*

A significant global I tells you clustering exists somewhere but not where. The
local statistics run at every unit.

```python
lisa = esda.local_moran(gdf["yield_bu_acre"].to_numpy(float), w_queen,
                        permutations=999, significance=0.05, seed=42)
gi   = esda.getis_ord_gi_star(gdf["yield_bu_acre"].to_numpy(float), w_queen,
                              permutations=999, significance=0.05, seed=42)

gdf["lisa_label"] = lisa.labels   # HH / LL / LH / HL / ns
gdf["gi_label"]   = gi.labels     # hot / cold / ns
```

Reading the LISA cluster map:

- `HH` and `LL` are cluster cores: a unit and its neighbours are both high (hot)
  or both low (cold).
- `LH` and `HL` are spatial outliers: a low unit ringed by high neighbours, or
  the reverse. They often sit at cluster edges or are genuinely anomalous.
- `ns` is everything that did not clear the significance threshold. Only units
  with `p_sim <= significance` keep a quadrant label; the rest are masked to
  `ns`. Mapping the unmasked quadrants would overstate the structure.

Getis-Ord Gi* answers a slightly different question: where are the local sums
unusually high or low? It uses `star=True`, so each unit is in its own
neighbourhood. Significant positive z-scores are hot spots, significant negative
ones cold spots, the rest `ns`. Gi* finds high/low pockets; LISA also separates
outliers from cluster cores. They tend to agree on the strong hot and
cold areas and to differ at the margins.

Run the whole pipeline at once and write a summary:

```bash
hotspots --config config/aoi.yaml --data data/raw/iowa_corn_yield_2023.gpkg --out outputs
```

This writes `outputs/esda_result.gpkg` (with the label columns) and
`outputs/summary.json` (global I and the per-label counts).

## 6. Optional: Geographically Weighted Regression

LISA and Gi* show where values cluster. GWR asks whether the relationship
between a response and covariates changes across space, by fitting a local
regression at each unit with nearby observations weighted more heavily.

```python
from hotspots import gwr

if gwr.mgwr_available():
    res = gwr.fit_gwr(coords, y, x, kernel="bisquare", fixed=False)
    print(res.bandwidth, res.aicc)
    gdf["local_r2"] = res.local_r2
```

GWR is exploratory. Local coefficients are correlated and the effective
parameter count is large, so plain t-tests overstate significance; use mgwr's
corrected critical values. Results depend on the kernel and bandwidth, so report
both. Local multicollinearity can flip a coefficient's sign in one region, so
read the coefficient surfaces rather than a single global number.

## 7. How to interpret responsibly

These statistics describe spatial pattern. They do not explain it, and a few
limits should travel with any result.

**They are not causal.** A hot spot says high values co-locate, not why. Soil,
climate, irrigation, and management are plausible drivers of a yield cluster but
none is tested here. Pattern is a prompt for explanation, not the explanation.

**The result depends on the weights.** Switch Queen to KNN to a distance band
and the clusters can move or disappear. Compare weights, report the neighbour
diagnostics, and check whether the headline finding survives a reasonable change
of W. A cluster that exists under only one definition of "neighbour" is fragile.

**They assume stationarity that may not hold.** A single global Moran's I treats
the entire map as one process. If the relationship genuinely varies across the
region, the global number averages over real local differences. That is the
reason GWR is offered as a follow-up.

**MAUP.** Conclusions are tied to the areal units and their scale. Counties,
tracts, and a regular grid can tell different stories from the same underlying
phenomenon (the modifiable areal unit problem). The choice of units is a result,
not a given.

**Edge effects.** Units at the boundary of the study area have neighbours
outside it that were never observed, so their local statistics rest on a
truncated neighbourhood and are less reliable than interior units.

**Multiple comparisons.** LISA and Gi* test every unit, so some will clear
`p = 0.05` by chance alone. The pseudo p-values are descriptive flags. Apply an
FDR or Bonferroni correction before making strong claims about any single unit,
and read the cluster map for coherent regions rather than isolated significant
cells.
