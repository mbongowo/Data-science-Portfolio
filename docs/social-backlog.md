# Social media content backlog

A standing queue of ready-to-post drafts mined from the 24 portfolio projects
plus five cross-cutting DevOps / platform themes. The weekly scheduler (see
`docs/social-automation.md`) pulls the next unused items from here, queues them
in Metricool **as drafts** for review, and marks them `scheduled`.

**Voice rules** (keep when editing): first person, plain and factual, specific
numbers over adjectives. No "dive in / unlock / game-changer / in today's world",
no em-dash overuse.

**Per-network split:** the **LinkedIn** copy is used for both LinkedIn and
Facebook. The **Bluesky** copy is a separate short post and MUST stay under 300
characters including any URL — re-check before scheduling.

**Status legend:** `unused` (available) · `scheduled` (queued in Metricool) ·
`posted` (published). Update the status line when an item is queued.

**Live demo URLs:**
- eo-explorer-app — https://data-science-portfolio-kpnqhpxmfzgwpgwbxkejql.streamlit.app/
- clinic-access dashboard — https://mbongowo-dat-spatial04-leafmap-dashboardappstreamlit-app-mclndk.streamlit.app/
- crop recommender — https://mbongowo-data-s-non-spatial04-ml-web-appappstreamlit-app-l5yxjk.streamlit.app/
- portfolio RAG — https://data-science-portfolio-fexfuaen4zdrzgpmu53nhs.streamlit.app/
- Repo — github.com/mbongowo/Data-science-Portfolio

> Note: the 8 launch posts in `docs/social-drafts.md` (anchor + 7 projects) were
> already scheduled Jun 23–Jul 8. This backlog is the *next* wave; some projects
> appear in both, written from a different angle.

---

## Spatial track

### eo-monitor — drought anomaly
Status: unused

**Facts:** Sentinel-2 L2A z-score anomalies vs a 4-year baseline; 34 spectral indices (NDVI/NDWI/NDMI + 31), pure numpy; SCL cloud masking; COG + PNG output; validated on the 2023 Corn Belt flash drought; demo recovers a planted loss patch at 1.0 recall, max |z|=42.12.

**LinkedIn:**
> How stressed was the 2023 Corn Belt flash drought? eo-monitor answers it by asking how far July–August greenness and canopy moisture departed from the 2019–2022 normal.
>
> The pipeline runs from a STAC catalogue (Sentinel-2 L2A, cloud-masked, pure-numpy indices) and produces per-pixel z-score anomaly maps. Negative z is stress; positive is greener or wetter than history. The demo recovers a planted loss patch at 1.0 recall with a peak z-score of 42.12.
>
> 34 spectral indices, z-score against a climatological baseline, cloud-optimised GeoTIFF output. Swap the config to query any field, year, or region.
>
> #geospatial #remotesensing #drought

**Bluesky:**
> Sentinel-2 z-score anomaly: how far did a field depart from its 4-year normal? Demo recovers planted loss exactly (1.0 recall, |z|=42.12). Config-driven, 34 indices, COG output.
> github.com/mbongowo/Data-science-Portfolio

### access-to-care — clinic equity
Status: unused

**Facts:** Multi-source Dijkstra travel time to nearest health facility over an OSM road network; WorldPop 100 m, Healthsites, GADM; population-weighted Gini; demo (Cameroon, 107,696 pop) 10% within 30 min, 35.1% within 60, 93.3% within 120, Gini 0.258; pure-python routing testable without a geo stack.

**LinkedIn:**
> How far is the nearest clinic? access-to-care measures it: travel time from every populated place to the nearest health facility over a real road network, weighted by population.
>
> Multi-source Dijkstra routes all cells to their nearest facility using highway-derived speeds. The output is the fraction of people who reach care within 30, 60, or 120 minutes. On the Cameroon demo (107k population): 10% within 30 min, 35% within 60, 93% within 120.
>
> The routing and equity arithmetic are pure Python, so the logic is fully testable without geospatial libraries. Outputs: per-admin coverage bands, a national summary, a population-weighted Gini, and facility load.
>
> #healthgeography #spatialanalysis #cameroon

**Bluesky:**
> Multi-source Dijkstra: what % of Cameroon reaches a clinic within 30/60/120 min? Demo: 10%/35%/93%. Pure-numpy routing over OSM + WorldPop + Healthsites.
> github.com/mbongowo/Data-science-Portfolio

### spatial-hotspots — does the statistic catch it?
Status: unused

**Facts:** Moran's I, Geary's C, LISA, Getis-Ord Gi*; pure-numpy core with hand-derived known-answer tests (monotone path I=1/3, checkerboard I=-1); demo on 12×12 grid: Moran's I 0.7242, Geary's C 0.2392, LISA 48 HH / 37 LL, Gi* 16 hot / 16 cold; pysal/esda lazily imported for permutation inference; USDA NASS yield default data.

**LinkedIn:**
> When values cluster in space, do these statistics actually catch it? spatial-hotspots tests that on a small synthetic grid: plant a high block and a low block in noise, add contiguity neighbours, and check whether Moran's I spikes, LISA flags the clusters, and Getis-Ord Gi* finds the hot and cold pockets.
>
> It does. Moran's I jumps to 0.72, Geary's C drops to 0.24, LISA recovers both planted cores (48 HH, 37 LL), and Gi* localizes 16 hot and 16 cold cells.
>
> The pure-numpy core (no geo stack) computes global Moran's I, Geary's C, local Moran, Getis-Ord Gi*, join counts, and bivariate Moran, each with hand-derived tests. The pysal wrappers add permutation inference for real data.
>
> #spatialstatistics #esda #gis

**Bluesky:**
> Plant a cluster in synthetic noise, run LISA / Getis-Ord. Does it catch it? Yes: Moran's I=0.72, Geary C=0.24. Pure-numpy core, hand-checked formulas, optional pysal permutation test.
> github.com/mbongowo/Data-science-Portfolio

### geoai-segmentation — reproducible by construction
Status: unused

**Facts:** U-Net (ResNet-34, Dice+BCE); reproducibility stack pixi.lock + Hydra resolved-config logging + seeded RNG + git SHA to MLflow; demo (seed 0): mean IoU 0.4778, pixel acc 0.8462, kappa 0.5215; pure-numpy metric/tiling core tested without torch; SpaceNet / Google Open Buildings data.

**LinkedIn:**
> Building-footprint segmentation is the easy part. The engineering problem is reproducibility: same seed, same config, same code, same numbers, every time.
>
> geoai-segmentation pins that with pixi (locked deps), Hydra (resolved config logged), seeded randomness across python/numpy/torch, MLflow (git SHA per run), and a tiling + metric core that runs on pure numpy with no GPU.
>
> The demo produces mean IoU 0.4778, pixel accuracy 0.8462, Cohen's kappa 0.5215 — identical on a re-run. Real training uses SpaceNet building footprints (free, AWS Open Data); the metric and tiling logic are unit-tested on CPU without a model.
>
> #computervision #reproducibility #mlops

**Bluesky:**
> U-Net segmentation: demo IoU 0.4778, same seed → same numbers. Pixi + Hydra + MLflow log the resolved config and git SHA. Pure-numpy metrics, reproducible on CPU.
> github.com/mbongowo/Data-science-Portfolio

### disturbance-detection — you need the time series
Status: unused

**Facts:** harmonic seasonal-trend fit + CUSUM breakpoint on NDVI residual; validated on the 2020 Creek Fire (breakpoint within 2 weeks of 2020-09-04 ignition, low false alarms); demo recovers a planted step at index 71, magnitude -0.099; Theil-Sen + Mann-Kendall; HLS/Landsat STAC cube on a 16-day grid, gaps are NaN not 0.

**LinkedIn:**
> One before/after image can't separate a real, persistent disturbance from a cloud shadow or a passing dry spell. You need the time series. disturbance-detection fits trend and seasonality to multi-year NDVI, then flags breakpoints.
>
> Harmonic regression per pixel, CUSUM on the residual. On the 2020 Creek Fire it places the breakpoint within two weeks of ignition across the burn perimeter, with sharp negative NDVI where the fire burned hottest and near-zero false alarms outside it.
>
> Pure-numpy core: harmonic decomposition, CUSUM breakpoint, recovery time, Theil-Sen slope, Mann-Kendall trend test. The demo recovers a planted step drop exactly. The cube is built from HLS or Landsat via STAC; gaps stay NaN, never 0.
>
> #changedetection #timeseries #remotesensing

**Bluesky:**
> CUSUM breakpoint detection on harmonic NDVI: on the 2020 Creek Fire it lands within 2 weeks of ignition. Demo recovers a planted step exactly. Pure-numpy core, HLS/Landsat STAC cube.
> github.com/mbongowo/Data-science-Portfolio

### eo-explorer-app — the library becomes the product
Status: unused

**Facts:** draw AOI → pick date (±10 days, least-cloudy) → pick index (34, 6 categories) → render on folium + download georeferenced GeoTIFF; reuses eo-monitor's index functions (not copy-pasted); Earth Search Sentinel-2 L2A, no auth; demo AOI 79.04 km², NDVI mean 0.6135; pure helpers (AOI validation, percentile stretch, deterministic cache keys); Streamlit Community Cloud.

**LinkedIn:**
> Draw an area on a map, pick a date, pick a spectral index. The app pulls live Sentinel-2 and renders NDVI, NDWI, NDMI, or 31 others on an interactive map, then hands you a georeferenced GeoTIFF.
>
> eo-explorer-app reuses the spectral-index math from eo-monitor rather than reimplementing it, so the analysis library becomes the engine inside a shipped product. Earth Search supplies Sentinel-2 L2A with no auth required.
>
> Pure-python helpers validate the AOI, compute robust statistics, and cache by (bbox, date, index). The downloaded GeoTIFF carries CRS and transform from the source, so it opens straight in QGIS or rasterio.
>
> Live: https://data-science-portfolio-kpnqhpxmfzgwpgwbxkejql.streamlit.app/
>
> #remotesensing #streamlit #geospatial

**Bluesky:**
> Draw an AOI, pick a date, pick an index (34 available), get a GeoTIFF. A Streamlit app that reuses eo-monitor's index math.
> Live: https://data-science-portfolio-kpnqhpxmfzgwpgwbxkejql.streamlit.app/

---

## Big data track

### clickstream-pipeline — windows as math
Status: unused

**Facts:** seeded demo 659 events / 200 users / 200 sessions (1800s gap); funnel view 100% → search 82% → cart 31% → checkout 18% → purchase 10.5%; peak 13 events/min (60s tumbling); pure-python reference tested with known-answer tests, same logic on Kafka + Spark Structured Streaming; explicit late-data handling (watermark, allowed lateness, dropped count).

**LinkedIn:**
> Stream processing is usually sold as infrastructure. I treated the logic as math: tumbling windows, sessionization, and funnel conversion all have boundary conditions (late events, session edges, dropped data) that belong in a test suite, not in production surprises.
>
> clickstream-pipeline has three layers — pure Python for precision, pandas for offline validation, Kafka + Spark for scale — running the same windowing logic. On a seeded 659-event stream, users drop from view (100%) to purchase (10.5%), peaking at 13 events/min.
>
> Late and out-of-order events are handled explicitly: a watermark, an allowed-lateness bound, and a reported count of what was dropped.
>
> #dataengineering #streaming #kafka

**Bluesky:**
> Clickstream funnel: view → search → cart → checkout → buy drops 100% → 10.5%. Pure-Python reference + Spark at scale, same logic both ways, late events handled explicitly.
> github.com/mbongowo/Data-science-Portfolio

### log-anomaly — honest about the threshold
Status: unused

**Facts:** 300 synthetic HDFS-like block sessions, 15% anomalous; Drain-lite templating → 8 templates; PCA reconstruction error (k=3) at 0.85 quantile; precision 0.74, recall 0.69, F1 0.71 (TN 244, FP 11, FN 14, TP 31 of 45); pure-numpy core (templating, event-count matrix, PCA/Mahalanobis/z-score detectors).

**LinkedIn:**
> Anomaly detection without labels works, but it is imperfect, and that is worth saying plainly.
>
> log-anomaly templates HDFS logs into 8 event types, builds an event-count matrix per block session, and scores each with PCA reconstruction error. Flagging blocks above the 85th percentile caught 31 of 45 real anomalies at 74% precision and 69% recall.
>
> The 11 false positives and 14 misses are not bugs. A tighter threshold kills recall, a looser one kills precision. The threshold is a business choice, not a tuning dial. Pure-numpy core, Spark at scale, every boundary case hand-verified in tests.
>
> #anomalydetection #dataengineering #machinelearning

**Bluesky:**
> HDFS log anomaly detection: PCA on templated event counts, 74% precision, 69% recall. The threshold drives the precision/recall trade — no magic value exists.
> github.com/mbongowo/Data-science-Portfolio

### als-recommender — measure honestly
Status: unused

**Facts:** MovieLens-25M (25M ratings, ~162k users, ~62k movies); demo 200×100, true rank 3, 20% holdout; RMSE popularity baseline 0.3612 vs ALS rank-6 0.1917; top-10 ALS Precision 0.0833 / Recall 0.8333 / NDCG 0.4777, baseline 0 on ranking; pure-numpy ALS variants + popularity baseline.

**LinkedIn:**
> Matrix factorization beats popularity, but only once you measure honestly.
>
> On MovieLens-25M, personalized ALS cuts RMSE from 0.36 to 0.19 against a "recommend the most popular movies" baseline. On ranking metrics (Precision@10, Recall@10, NDCG@10) the baseline scores zero, because globally popular titles are not the ones each user already rated highly.
>
> A seeded synthetic demo (200 users, 100 items, true rank 3) keeps the numbers reproducible. Pure-numpy reference, Spark MLlib for scale. The caveats stay on the label: cold start falls back to popularity, and offline lift is not online engagement.
>
> #recsys #machinelearning #dataengineering

**Bluesky:**
> ALS on MovieLens-25M: RMSE 0.1917 vs 0.3612 popularity baseline. On ranking metrics the baseline scores 0 — popular ≠ relevant per user. Pure numpy + Spark MLlib.
> github.com/mbongowo/Data-science-Portfolio

### sentiment-scale — validate before you quote a number
Status: unused

**Facts:** lexicon scorer (valence + 3-token negation window + squash) sign accuracy 1.00 on 336 labelled posts; planted shift recovered: weekly mean +0.28 (pre) → -0.37 (post), 0.58 swing; LogisticRegression bag-of-words alternative; lexicon catches negation but not sarcasm; Spark + NLP on real Reddit dumps.

**LinkedIn:**
> A sentiment-over-time chart can be signal or noise. The difference is whether you validated it.
>
> sentiment-scale scored 336 labelled posts with a VADER-style lexicon at 100% sign accuracy. When I planted a sentiment shift in the corpus, the lexicon recovered it: the weekly mean flipped from +0.28 before to -0.37 after, a 0.58-point swing.
>
> But lexicons miss context — sarcasm reads positive, topic drift confounds the trend. So the project ships a trained alternative (logistic regression on bag-of-words), reports both scorers, and checks topic clusters alongside the trend. Pure-numpy core, Spark + NLP at scale.
>
> #nlp #sentimentanalysis #dataengineering

**Bluesky:**
> Reddit sentiment: lexicon vs trained model on a planted shift. The lexicon caught a +0.28 → -0.37 swing at 100% sign accuracy on labelled posts. Validate before you quote a number.
> github.com/mbongowo/Data-science-Portfolio

### tlc-analytics — engine bake-off
Status: unused

**Facts:** NYC TLC yellow taxi, billions of partitioned Parquet rows; projected single-machine runtime DuckDB 42s / Spark 110s / warehouse 18s; peak demand 18:00, mean fare $26.49, card tip 17.9% vs cash 0.0% (meter doesn't log cash); pure pandas/numpy reference with Tukey-fence outliers.

**LinkedIn:**
> Which engine should run your analytical workload? It depends on your constraints, so I measured.
>
> The same aggregation (demand by hour, tips by payment type, fare summaries) over NYC TLC Parquet, through three engines: DuckDB (42s), Spark (110s), a managed warehouse (18s, billed per query). On a single machine, DuckDB's in-process vectorisation beats JVM overhead; Spark pulls ahead once the lake outgrows one box; the warehouse costs nothing upfront but bills per scan.
>
> One finding worth flagging: card tipping averages 17.9% and cash shows 0% — not because riders don't tip cash, but because the meter never logs it. A measurement artifact, not behaviour.
>
> #dataengineering #duckdb #spark

**Bluesky:**
> TLC taxi engine bake-off: DuckDB 42s, Spark 110s, warehouse 18s (single machine). Peak demand 18:00; card tips 17.9%, cash logged as 0% (meter doesn't record it).
> github.com/mbongowo/Data-science-Portfolio

### dbt-modern-stack — test before the warehouse
Status: unused

**Facts:** pure-pandas demo: 13 tests pass on clean data, planted null/dup/orphan break exactly 5; full dbt on IMDb in DuckDB: 31 tests (28 generic + 1 singular + 2 freshness); generic tests mirrored in pandas (not_null, unique, accepted_values, relationships, range, freshness); heavy deps lazily imported.

**LinkedIn:**
> Data-quality tests should run before you have a warehouse.
>
> dbt-modern-stack builds a dbt project on IMDb data with generic tests (not_null, unique, accepted_values, relationships) and mirrors the same test logic in pure pandas. CI runs the pandas version in seconds; the warehouse version runs the same tests on real data after the models build.
>
> The payoff is being able to test the logic before standing up DuckDB or BigQuery. Plant a null and exactly the not_null tests fail; plant a duplicate and the unique tests fail. Same row-level expression test catches business-rule violations.
>
> #dbt #dataengineering #testing

**Bluesky:**
> dbt + pure pandas: test data-quality logic before the warehouse exists. 13 generic tests mirrored in numpy/pandas, pass in CI, then run in dbt at scale on real IMDb data.
> github.com/mbongowo/Data-science-Portfolio

### crypto-backtest — rigour over returns
Status: unused

**Facts:** SMA(10/30) crossover on 1000 one-minute bars (60k synthetic ticks); total return -3.50%, Sharpe -3.82, max drawdown 12.13%, 36 trades, 15 bps cost per change; buy-and-hold +1.26%; signal on bar t executes on bar t+1 (no look-ahead); data-integrity checks first.

**LinkedIn:**
> A backtest number is only credible if you can defend every assumption behind it.
>
> crypto-backtest runs an SMA(10/30) crossover on synthetic one-minute bars: -3.50% total return, Sharpe -3.82, 12% max drawdown, against buy-and-hold at +1.26%. The strategy lost money, and that is the honest result — the synthetic path has zero drift by design, so a costed crossover is expected to lose.
>
> The point isn't the return, it's the machinery: signals on bar t execute on bar t+1 (no look-ahead), costs are explicit at 15 bps per trade, and data integrity (gaps, duplicates, coverage) is checked before anything runs. The same pipeline runs on real Binance ticks via Polars/Spark.
>
> #backtesting #quant #dataengineering

**Bluesky:**
> Backtest SMA crossover on 1000 bars: -3.50% return, 12% max drawdown, zero look-ahead, 15 bps costs. Buy-and-hold did +1.26%. The rigour matters more than the number.
> github.com/mbongowo/Data-science-Portfolio

### graph-analysis — recover the planted structure
Status: unused

**Facts:** seeded SBM 30 nodes / 83 edges / 3 communities; PageRank top [16,11,6,4,10]; label propagation exact recovery of 3 communities; 69 triangles, clustering ~0.505, betweenness top node 16 (= PageRank top); modularity Q 0.559; LiveJournal scale 4.8M nodes / 69M edges on Spark GraphFrames.

**LinkedIn:**
> When label propagation recovers exactly the communities you planted, you know the machinery works.
>
> graph-analysis builds a seeded stochastic block model with 30 nodes and 3 planted communities. Label propagation finds exactly 3. PageRank and betweenness centrality both flag the same hub node. Modularity on the planted partition is 0.559, versus 0 for a single community.
>
> On real networks — LiveJournal, 4.8M nodes and 69M edges — the same four algorithms run on Spark GraphFrames. The interpretation comes with caveats: PageRank rankings shift with the damping factor, label propagation has a resolution limit, and triangle counts measure clustering, not causation.
>
> #graphs #networkanalysis #dataengineering

**Bluesky:**
> Graph demo: 30 nodes, 3 planted communities. Label propagation recovers them exactly; PageRank's top node matches betweenness centrality's. Pure numpy + Spark GraphFrames at 4.8M-node scale.
> github.com/mbongowo/Data-science-Portfolio

---

## Technique-replication track

### 01-segment-geospatial — masks into numbers
Status: unused

**Facts:** built on opengeos/segment-geospatial; adds pure-numpy quantification (connected components, region props, area filtering) with known-answer tests; Douala AOI; demo 25 buildings (mean 26.77 m², total 669.25 m²), 1 field 0.0575 ha, geometry round-trips at IoU 1.000; config-driven YAML.

**LinkedIn:**
> Segment Anything can segment buildings in a satellite basemap with no training, but a raw mask is just pixels. It doesn't tell you how many buildings there are or how big they are.
>
> 01-segment-geospatial adds the step the original repo leaves out: a pure-numpy quantification stage between a SAM mask and reportable numbers — connected-component labelling, per-polygon areas, size filtering to separate buildings from shadows, and conversion to square metres and hectares.
>
> Every function has hand-derived tests. The Douala demo recovers 25 buildings at a mean footprint of 26.77 m², geometry round-tripping at IoU 1.000. Define your region once in YAML and it runs anywhere.
>
> #geospatial #computervision #cameroon

**Bluesky:**
> A post-processor for Segment Anything: SAM masks → counted, measured building footprints. Douala demo recovers 25 buildings (mean 26.77 m²) at IoU 1.0. Pure numpy, fully tested.
> github.com/mbongowo/Data-science-Portfolio

### 02-earth-engine-timeseries — auth-free forest change
Status: unused

**Facts:** inspired by giswqs/geemap; Earth-Engine-free, pulls Sentinel-2 L2A from open Earth Search STAC (no auth); pure-numpy multi-year NDVI composites, bitemporal change, dNBR; Cameroon forest AOI; demo baseline NDVI 0.8501 → recent 0.7655, 9 ha planted clearing recovered at recall 1.000; validated vs Global Forest Watch / Hansen.

**LinkedIn:**
> Forest loss in the Congo Basin happens in patches that are easy to miss between annual reports, and a single satellite date is too cloud-prone over equatorial forest to be reliable.
>
> 02-earth-engine-timeseries builds a cloud-free baseline composite and a recent one, differences them to map loss, and totals the hectares. The twist on the geemap original: it pulls multi-year Sentinel-2 from an open STAC catalogue with no authentication, and processes it with a pure-numpy core (median composites, NDVI change, loss/gain classification).
>
> The demo recovers a planted 9 ha clearing perfectly, and results are validated against Global Forest Watch / Hansen tree-cover loss. Earth Engine stays an optional path once you're ready to authenticate.
>
> #forestmonitoring #remotesensing #cameroon

**Bluesky:**
> Sentinel-2 forest-change: baseline vs recent NDVI composites → loss/gain → hectares. Demo recovers a 9 ha clearing perfectly (recall 1.0). Validated vs Global Forest Watch. Auth-free STAC, no Earth Engine needed.

### 03-torchgeo-landcover — pretrained vs from scratch
Status: unused

**Facts:** inspired by microsoft/torchgeo; compares ImageNet-pretrained ResNet18 vs from-scratch on EuroSAT; pure-numpy softmax baseline reproducible without GPU; demo 480 patches / 6 classes: accuracy 0.885, macro-F1 0.882, kappa 0.863, top-2 1.000; same metrics code for both models; runs in <2s.

**LinkedIn:**
> How much does ImageNet pretraining actually buy you on a small remote-sensing dataset, versus training from scratch? 03-torchgeo-landcover answers it measurably.
>
> I built a pure-numpy softmax baseline (multinomial logistic regression on per-band features) that trains and validates in under a second with no GPU — the part that decides what a number means is reproducible on any machine. Then I fine-tuned a TorchGeo ResNet18 on EuroSAT, pretrained and from scratch, using the same metrics code so the comparison is honest.
>
> The baseline reaches 0.885 accuracy and 0.863 Cohen's kappa on held-out test, with the metrics pinned by tests so they stay comparable.
>
> #machinelearning #transferlearning #remotesensing

**Bluesky:**
> ImageNet pretraining vs from-scratch on EuroSAT, judged by the same metrics. Pure-numpy baseline hits 0.885 acc / 0.863 kappa; TorchGeo ResNet18 is the deep comparison. Demo runs in <2s, no GPU.

### 04-leafmap-dashboard — coverage gaps you can click
Status: unused

**Facts:** built on opengeos/leafmap; haversine nearest-facility distance fast enough to recompute per slider move; real data ~2,070 OSM facilities × ~1,290 places, mean nearest 21.1 km, median 15.9 km; Bétaré-Oya ~115 km from a mapped clinic; live on Streamlit Community Cloud; upload custom CSVs or edit config.

**LinkedIn:**
> Health planners need a quick, honest first pass at coverage gaps: given where people live and where clinics are, which settlements are worst served?
>
> 04-leafmap-dashboard answers that interactively. Pick an "underserved" distance threshold, watch the map highlight places beyond it, export the ranked list, and drill down by region or division. The core is deliberately simple — straight-line haversine distance to the nearest facility, fast enough to recompute on every slider move — which makes it a practical screening tool.
>
> It's fully data-driven: upload your own CSVs or use the bundled real Cameroon data (~2,070 OSM facilities, ~1,290 places). On that data the farthest mapped town, Bétaré-Oya, sits ~115 km from a clinic, subject to OSM completeness.
>
> Live: https://mbongowo-dat-spatial04-leafmap-dashboardappstreamlit-app-mclndk.streamlit.app/
>
> #healthdata #gis #cameroon

**Bluesky:**
> Interactive clinic-access dashboard: pick a distance threshold, see underserved places on a map, export the ranked list. Real Cameroon data (~2,070 facilities, ~1,290 places).
> Live: https://mbongowo-dat-spatial04-leafmap-dashboardappstreamlit-app-mclndk.streamlit.app/

### 05-change-detection — SAR sees through clouds
Status: unused

**Facts:** inspired by robmarkcole/satellite-image-deep-learning; uses Sentinel-1 SAR (floods come with clouds); Otsu auto-threshold per date, before/after water masks, flooded/permanent/receded, hectares; Logone-et-Chari floodplain; demo recovers a 12 ha flood at recall 0.999; validated vs UN OCHA 2024 (156,000 people, 82,509 ha farmland).

**LinkedIn:**
> Cameroon's Far North floods almost every rainy season, displacing tens of thousands and drowning cropland. The 2024 event alone affected about 156,000 people.
>
> Mapping floods with optical sensors fails exactly when you need them, because the clouds are there too. SAR solves it: radar penetrates cloud, and smooth water reflects it away, so flooded ground shows up dark.
>
> 05-change-detection is the pipeline from open Sentinel-1 to a flooded-hectares number. The pure-numpy core converts to dB, applies Otsu thresholding per date (no manual tuning), builds before/after water masks, and totals hectares by class. The demo recovers a planted 12 ha flood at 0.999 recall, validated against UN OCHA situation reports.
>
> #floodmonitoring #SAR #cameroon

**Bluesky:**
> SAR flood mapping: Sentinel-1 before/after → Otsu threshold → water masks → flooded hectares. Demo recovers a 12 ha flood at 0.999 recall. Validated against UN OCHA reports. Auth-free STAC.

### 01-data-engineering-pipeline — the boring plumbing, done right
Status: unused

**Facts:** capstone shape from DataTalksClub DE zoomcamp; Open-Meteo → partitioned Parquet lake → DuckDB or Azure SQL → dbt marts → Streamlit; demo 1,823 records / 5 Cameroon stations, 3 planted bad rows rejected (99.84% valid), 28 dbt tests; Terraform Azure path opt-in; Prefect orchestration; pure-pandas core runs in <1s.

**LinkedIn:**
> Cameroon's climates differ enough — wet coastal Douala, semi-arid Maroua — that a single "Cameroon weather" number is meaningless. You need clean, tested, region-specific data before any analysis can be trusted.
>
> 01-data-engineering-pipeline is the plumbing for that: ingest historical weather from Open-Meteo, land it in a partitioned lake, load it to a warehouse (DuckDB by default, Azure SQL via Terraform), transform with dbt and run 28 data-quality tests, then surface it to a Streamlit dashboard.
>
> The reproducible core is pure pandas: the demo runs 1,823 records through validation, rejects 3 planted bad rows (tmin > tmax, negative precip, duplicate key) at 99.84% valid, in under a second. The Azure path is opt-in and nothing deploys without an explicit decision.
>
> #dataengineering #dbt #cameroon

**Bluesky:**
> End-to-end weather pipeline: Open-Meteo → Parquet lake → DuckDB warehouse → dbt marts (28 tests) → dashboard. Demo: 1,823 records, 5 Cameroon cities, 99.84% valid. Pure pandas, runs in <1s.

### 02-mlops-pipeline — the part after "trained a model"
Status: unused

**Facts:** inspired by DataTalksClub MLOps zoomcamp; rain-day prediction from 12 features; demo accuracy 0.8939, F1 0.6667, ROC-AUC 0.9379; drift report flags 10 of 12 features (max PSI 8.79) in a warmer/wetter regime, leaves seasonal encoding untouched; pure-numpy ML + drift core; MLflow / FastAPI / Evidently as lazy wrappers.

**LinkedIn:**
> Most ML portfolios stop at "trained a model." The value is in what comes after deployment: noticing when the model has gone stale instead of silently serving bad predictions.
>
> 02-mlops-pipeline closes that loop. It predicts whether it rains tomorrow at a Cameroon station from 12 engineered features (lags, rolling means, seasonal encoding) and reaches 0.894 accuracy and 0.938 ROC-AUC on a time-ordered split. Then it synthesizes a warmer, wetter regime to simulate drift.
>
> The drift report (PSI + KS per feature) flags 10 of 12 features with a max PSI of 8.79, while correctly leaving the two seasonal features alone — exactly the selective signal you want before deciding to retrain. MLflow, FastAPI, and Evidently sit behind lazy imports so the tested core stays pure numpy.
>
> #mlops #driftdetection #machinelearning

**Bluesky:**
> Rain-day prediction + drift monitoring: 0.894 acc / 0.938 ROC-AUC baseline. The drift report catches 10 of 12 features in a warmer regime (max PSI 8.79) and correctly ignores seasonal encoding. The MLOps loop, closed.

### 03-streaming-pipeline — alert once, not 187 times
Status: unused

**Facts:** inspired by damklis/DataEngineeringProject; air-quality alerting for Cameroon cities; PM2.5/PM10 → EPA AQI, tumbling windows, per-(station,rule) cooldown; demo 384 readings / 4 stations / 4 days, 51 alerts fired vs 187 suppressed, peak AQI 173 (Unhealthy) at Garoua; pure-python core, Kafka+Spark behind lazy imports, Redpanda local.

**LinkedIn:**
> A monitoring system that alerts on every hour of a day-long pollution episode is worse than useless. The hard part isn't the threshold, it's not alert-storming.
>
> 03-streaming-pipeline streams hourly PM2.5/PM10 for Cameroon cities, converts to EPA AQI, and applies three rule types: threshold (WHO guideline), AQI category (Unhealthy band), and spike (3σ above a rolling mean). The differentiator is the alert engine: it remembers the last fire time per (station, rule) and suppresses repeats inside a cooldown window.
>
> The demo plants a Garoua dust episode that peaks at AQI 173. The engine fires once, then the cooldown absorbs the repeats: 51 alerts fired, 187 suppressed. The same pure-python logic runs on Kafka + Spark at scale; local dev uses a Redpanda broker.
>
> #streaming #airquality #kafka

**Bluesky:**
> Air-quality streaming + alerting: 4 Cameroon cities, EPA AQI, threshold/spike/category rules, per-(station,rule) cooldown. Demo: 51 alerts fired, 187 suppressed — no alert-storm. Runs in <1s.

### 04-ml-web-app — explainable crop choice
Status: unused

**Facts:** inspired by shsarv/Machine-Learning-Projects; soil (N,P,K) + climate (temp, humidity, pH, rainfall) → ranked top-3 crops; pure-numpy softmax trained at startup, optional sklearn RF on Kaggle data; demo 800 samples / 10 crops: accuracy 0.8958, macro-F1 0.8942, top-3 0.9875; Streamlit Community Cloud.

**LinkedIn:**
> Choosing what to plant turns on conditions a soil test and a weather record can measure: N, P, K, temperature, humidity, pH, and rainfall.
>
> 04-ml-web-app maps those seven numbers to a recommended crop with a ranked top-3 and confidence scores, so the choice is explainable rather than a single opaque verdict. The model is a pure-numpy softmax trained at startup on bundled data, so the app needs only numpy, pandas, and Streamlit and retrains itself on every redeploy.
>
> On a seeded dataset of ten Cameroon-relevant crops it reaches 0.896 accuracy and 0.988 top-3 accuracy — the right crop is almost always in the shortlist. It's decision support, not authoritative agronomy; real N/P/K values need a soil test.
>
> Live: https://mbongowo-data-s-non-spatial04-ml-web-appappstreamlit-app-l5yxjk.streamlit.app/
>
> #machinelearning #agriculture #cameroon

**Bluesky:**
> Crop recommender: soil (N,P,K) + climate (temp, humidity, pH, rainfall) → ranked top-3 crops. Pure numpy, 0.896 acc, 0.988 top-3. The app trains itself at startup.
> Live: https://mbongowo-data-s-non-spatial04-ml-web-appappstreamlit-app-l5yxjk.streamlit.app/

### 05-focused-ml-project — RAG that can't hallucinate by default
Status: unused

**Facts:** portfolio RAG; pure-python TF-IDF retrieval + MMR re-ranking; demo 14 docs / 29 chunks / 648 terms, recall@3 1.00, MRR 1.00 on 14 curated QA pairs; extractive by default (free, no key, no hallucination), optional OpenAI/Azure OpenAI; Streamlit Community Cloud.

**LinkedIn:**
> A portfolio of two dozen projects is hard to browse. The natural interface is a question with a cited answer.
>
> 05-focused-ml-project is a RAG system over my portfolio docs, and the contribution is a retrieval core that needs no hosted index and no API key: chunk each project's markdown, build an L2-normalised TF-IDF index, retrieve top-k by cosine similarity, then re-rank with Maximal Marginal Relevance to trade relevance against redundancy.
>
> On 14 docs (29 chunks, 648 terms) and 14 curated questions, recall@3 and MRR are both 1.00 — a small, well-separated corpus where each project has distinctive vocabulary. The default mode is extractive: it quotes the most relevant chunk and lists sources, so it can't hallucinate. Generated answers via OpenAI or Azure OpenAI are optional.
>
> Live: https://data-science-portfolio-fexfuaen4zdrzgpmu53nhs.streamlit.app/
>
> #rag #nlp #machinelearning

**Bluesky:**
> Portfolio RAG: TF-IDF retrieval (14 docs, 648 terms) + MMR re-ranking. Recall@3 1.00, MRR 1.00 on 14 curated Q&A pairs. Extractive by default (free, no key, no hallucination).
> Live: https://data-science-portfolio-fexfuaen4zdrzgpmu53nhs.streamlit.app/

---

## DevOps & platform track

### ci-cd — one pipeline, 24 projects
Status: unused

**Facts:** single `.github/workflows/ci.yml` on every push and PR; matrix across all 24 projects; Python 3.12 on Ubuntu; lightweight deps only (numpy, pandas, pyyaml, pydantic, typer, pytest, ruff); heavy ML/geo stacks excluded and guarded to skip; ruff lint + `pytest -q` per project; src/ and app-root on PYTHONPATH.

**LinkedIn:**
> One CI workflow guards all 24 projects in this portfolio. It runs on every push and pull request, and it tests each project against a deliberately small core: numpy, pandas, pyyaml, pytest, ruff. Nothing heavy.
>
> The decision behind that: I don't install PyTorch, TensorFlow, or GDAL in CI. The pure-numerical core of every project is tested, and anything needing a heavy engine is guarded so it skips. The pipeline stays fast and the failures it reports are real, not environment noise.
>
> Each project lints with ruff and runs pytest on Python 3.12, with a src/ layout so tests can't import the wrong thing.
>
> #devops #cicd #python

**Bluesky:**
> One CI workflow, 24 projects, every push. It tests the numpy/pandas core only; heavy ML and geo stacks are guarded and skip. Fast, and the failures are real. Python 3.12 + ruff.

### infrastructure-as-code — free by default, paid on purpose
Status: unused

**Facts:** Terraform under `01-data-engineering-pipeline/terraform/azure/` (Terraform >= 1.5, azurerm ~> 3.110); provisions resource group, ADLS Gen2 (hierarchical namespace), serverless Azure SQL (GP_S_Gen5_1, auto-pause 60 min), firewall rules, random suffix; sensitive password, no default; free local DuckDB path is the default; plan/validate free, apply costs, destroy cleans up.

**LinkedIn:**
> Every cloud-touching project in my portfolio has two paths: a free local one that's the default, and an opt-in Azure one behind Terraform.
>
> For the Cameroon weather pipeline the Terraform defines an Azure landing zone — ADLS Gen2 for the lake, serverless Azure SQL for the warehouse (it auto-pauses after 60 idle minutes so it isn't billing while you sleep), a resource group, firewall rules. None of it runs unless you apply it; the default pipeline runs on DuckDB locally with no bill.
>
> I treat that boundary carefully: `terraform plan` is free and shows exactly what would be created, `apply` is the only step that costs money, and `destroy` cleans up. The admin password has no default and must be passed in. Infrastructure shouldn't surprise you.
>
> #terraform #iac #azure

**Bluesky:**
> Cloud projects ship two paths: free local (DuckDB) by default, opt-in Azure via Terraform. The IaC provisions ADLS Gen2 + serverless SQL that auto-pauses. plan is free, destroy cleans up.

### containers — strategy by weight class
Status: unused

**Facts:** 17 Dockerfiles; python:3.12-slim for pure-numeric cores (~150 MB), pixi/conda-forge for full geo/ML stacks with committed pixi.lock (copy lock first, then source, for cache); 2 docker-compose stacks — MLflow + FastAPI (mlops), and Redpanda + producer + processor + Streamlit (streaming) with a broker healthcheck; lazy imports keep the inference image small.

**LinkedIn:**
> Two containerisation strategies across the portfolio, picked by weight class.
>
> For pure-numeric cores — the parts CI tests — I build from python:3.12-slim, so numpy + pandas + pytest lands around 150 MB. For full geospatial and ML stacks I use pixi over conda-forge, because it manages native bindings like GDAL far better than pip, and I commit pixi.lock so a build two years from now is identical. The Dockerfile copies the lock and resolves it before the source, so a code change doesn't re-download the heavy bindings.
>
> For stateful services I use docker-compose: the streaming project brings up a Redpanda broker, a producer, a windowing/alerting processor, and a Streamlit dashboard in one command; the MLOps project brings up MLflow and a FastAPI service.
>
> #docker #devops #python

**Bluesky:**
> 17 Dockerfiles, 2 compose stacks. Slim base for numeric cores (~150 MB), pixi + lock for full geo/ML stacks, docker-compose for stateful services (Redpanda, MLflow, FastAPI).

### reproducible-environments — designed against "works on my machine"
Status: unused

**Facts:** pixi + conda-forge across 14 projects (platforms win-64/linux-64/osx-64/osx-arm64); committed pixi.lock on production-scale projects; src/ layout; shared Makefile (test/lint/format/clean + demo targets); identical pre-commit hooks (ruff, mypy on src, whitespace, YAML, large-file check); CI mirrors with Python 3.12 + lightweight deps.

**LinkedIn:**
> "Works on my machine" is a bug I designed against. Every one of the 24 projects shares the same environment story, on purpose:
>
> - pixi + conda-forge, so Python and native bindings (GDAL, GEOS) come from one source and resolve identically on Windows, Linux, Intel Mac, and Apple Silicon.
> - a committed pixi.lock on the production-scale projects, so a Docker build is byte-for-byte identical across machines and years.
> - a src/ layout so tests can't accidentally import from the working directory.
> - a Makefile with the same targets everywhere: make test, make lint, make format.
> - the same pre-commit hooks (ruff, mypy, whitespace, YAML) in every folder.
>
> The payoff: clone any project, run pixi install then make test, and you get exactly what CI gets. No per-project setup notes.
>
> #devops #reproducibility #python

**Bluesky:**
> Designed against "works on my machine": pixi + pixi.lock + src/ layout + a shared Makefile + identical pre-commit hooks across 24 projects. Clone → pixi install → make test = what CI runs.

### mlops-platform — a platform without Kubeflow
Status: unused

**Facts:** MLflow tracking (local SQLite or Azure ML workspace via one env var), FastAPI `/predict` service, PSI + KS drift in pure numpy (no SciPy), optional Evidently dashboard; lazy-import wrapper pattern keeps the tested core at numpy/pandas; `mlpipe` CLI (train/serve/monitor/demo); docker-compose to bring it up.

**LinkedIn:**
> An MLOps "platform" doesn't have to mean Kubeflow and a dedicated team. The rain-day project does it with three tools and one design decision.
>
> Training runs (params, metrics, the fitted model) log to MLflow — the same code points at a local SQLite backend or an Azure ML workspace, switched by one env var. A FastAPI service serves the model at /predict. Both come up with docker compose.
>
> The part most portfolios skip is what happens after deploy: drift. I compute PSI and the KS statistic on each incoming batch, in pure numpy, no SciPy; PSI >= 0.2 flags a feature for retraining, and an optional Evidently dashboard renders the report. The design decision that holds it together is lazy imports — the tested core pulls in only numpy/pandas, and MLflow, FastAPI, and Evidently load only when called. CI tests drift detection without ever importing the serving stack.
>
> #mlops #python #machinelearning

**Bluesky:**
> MLOps without Kubeflow: MLflow tracking + FastAPI serving + PSI/KS drift in pure numpy. Lazy imports keep the tested core at numpy/pandas; heavy deps load only when called. Local SQLite or Azure ML, one env var.

---

## DevOps repo track (github.com/mbongowo/DEVOPS)

Mined from the standalone DevOps monorepo — production-shaped CI/CD, GitOps,
Kubernetes, observability, and IaC, most of it validated offline/locally for
zero cloud cost. Repo link for Bluesky: github.com/mbongowo/DEVOPS

### DEVOPS/01-three-tier-devsecops-eks — code to self-healing on EKS
Status: unused

**Facts:** three-tier app (React / Node / MongoDB) on AWS EKS; Terraform (VPC, EKS, ECR); GitHub Actions CI runs npm test then Trivy image scan (CRITICAL gate) + optional SonarQube gate; Helm deploy; Argo CD GitOps reconciliation; ingress + cert-manager TLS; Prometheus/Grafana.

**LinkedIn:**
> I built an end-to-end workflow for a three-tier app on AWS EKS: React frontend, Node backend, MongoDB, from a git push to a monitored, self-healing deployment.
>
> The pipeline runs on every push: npm test and build, then Trivy scans the images and blocks fixable CRITICAL CVEs, with an optional SonarQube quality gate. If tests break, the image never gets built.
>
> Infrastructure is Terraform (VPC, security groups, EKS, ECR), validated offline in CI so nothing deploys without a working plan. Argo CD then takes over with Git as the source of truth, and the cluster self-heals on drift. Prometheus and Grafana sit on top so failures surface before users hit them.
>
> #devops #kubernetes #terraform #cicd #gitops

**Bluesky:**
> Built a three-tier app on EKS: React/Node/MongoDB with GitHub Actions CI (test → SonarQube → Trivy scan), Terraform IaC, Argo CD GitOps, Prometheus/Grafana. Shift-left security + declarative infra.
> github.com/mbongowo/DEVOPS

### DEVOPS/02-robotshop-observability — SLOs and burn-rate alerts
Status: unused

**Facts:** polyglot microservices (Node, Java, Python, Go, PHP) on Kubernetes; Prometheus + Alertmanager + Grafana; OpenTelemetry → Tempo traces, Promtail → Loki logs; SLOs + multi-window burn-rate alerts (Google SRE method); alert rules unit-tested with promtool; golden-signals dashboards; k6 load + injected-fault postmortem; runs on a local kind cluster, zero cost.

**LinkedIn:**
> I built the observability layer for Stan's Robot Shop, a polyglot microservices app, to show how monitoring works past "is it up?".
>
> Prometheus scrapes metrics, Alertmanager routes them, Grafana renders dashboards as code. OpenTelemetry flows traces into Tempo and Promtail ships logs to Loki, so a request is traceable end to end.
>
> The core is SLOs with burn-rate alerts: a fast-burn alert fires on a sharp error spike, and a slow-burn alert catches a gradual one before the error budget is gone. Every alert rule is unit-tested with promtool, so a refactor that breaks an alert fails CI. The whole thing runs on a local kind cluster, and I wrote a postmortem for an injected fault.
>
> #observability #prometheus #grafana #sre

**Bluesky:**
> Observability for Robot Shop: Prometheus metrics, Tempo traces, Loki logs, Grafana dashboards. SLOs + burn-rate alerts (Google SRE method), promtool-tested rules, k6 load + incident postmortem. Local kind, zero cost.
> github.com/mbongowo/DEVOPS

### DEVOPS/03-sockshop-gitops — Prometheus-gated canary
Status: unused

**Facts:** Sock Shop via GitOps; Argo CD App-of-Apps + AppProject guardrails; ApplicationSet generates per-env (dev/prod) Applications; Kustomize base + overlays; Argo Rollouts canary shifts 20% traffic and proceeds only if Prometheus success rate ≥ 95%, else auto-rollback; drift correction; offline manifest validation with kubeconform.

**LinkedIn:**
> GitOps means Git is the only place you edit: not kubectl apply, not helm install, just a git push.
>
> I set up Argo CD to manage Sock Shop across dev and prod from a single ApplicationSet template, one list entry per environment. The frontend deploys as an Argo Rollouts canary that shifts 20% of traffic, polls Prometheus for the success rate, and only proceeds if it clears 95%, reverting to stable otherwise.
>
> Every manifest is offline-validated before it reaches the cluster: Kustomize renders it, kubeconform schema-checks it. And if someone scales a deployment by hand, Argo CD catches the drift and reverts it.
>
> #gitops #argocd #kubernetes #devops

**Bluesky:**
> GitOps for Sock Shop: Argo CD App-of-Apps + ApplicationSet per environment, Kustomize overlays, Argo Rollouts canary with auto-rollback gated on a Prometheus 95% success rate. Git is the source of truth.
> github.com/mbongowo/DEVOPS

### DEVOPS/04-devsecops-cicd-pipeline — five hard security gates
Status: unused

**Facts:** GitHub Actions with five blocking gates — gitleaks (committed secrets), Semgrep SAST (p/default), Trivy filesystem (HIGH/CRITICAL), Trivy image (fixable CRITICAL), Trivy IaC config (HIGH/CRITICAL misconfig); Checkov as a second IaC opinion; optional SonarQube + OWASP Dependency-Check; all reports uploaded as artifacts; Flask app + hardened S3 Terraform as targets.

**LinkedIn:**
> Shift-left security means scanning in the pipeline, so regressions are caught before code reaches review rather than waiting on a pentest.
>
> I wired five blocking gates into a GitHub Actions pipeline: gitleaks for committed secrets, Semgrep SAST for injection and unsafe crypto, Trivy across the filesystem (high/critical CVEs and secrets), the container image (fixable criticals), and the IaC (misconfigurations like a public S3 bucket). Each gate writes a JSON report uploaded as an artifact, and any failure stops the build before the image is pushed.
>
> SonarQube and OWASP Dependency-Check are optional add-ons behind secrets, but the core gates run free and offline, so the same commands reproduce locally.
>
> #devsecops #cicd #security #devops

**Bluesky:**
> Shift-left security pipeline: gitleaks (secrets), Semgrep (SAST), Trivy (filesystem / image / IaC), Checkov (IaC). Five hard gates, all free and offline; each blocks on findings and uploads a report.
> github.com/mbongowo/DEVOPS

### DEVOPS/05-flask-cicd-terraform — local pipeline that maps to EKS
Status: unused

**Facts:** Flask URL-shortener (shorten API, health + metrics); GitHub Actions CI: flake8 + pytest, docker build + container smoke test; kind-based end-to-end smoke test (Terraform + Helm → deploy → live curl); Terraform uses the kind provider locally and maps to AWS EKS; Helm chart; Prometheus /metrics + Grafana dashboard; Jenkinsfile alternative; zero cloud cost.

**LinkedIn:**
> Most "production pipeline" walkthroughs assume you already pay for AWS or GKE. This one runs on a local kind cluster for free, then maps cleanly to EKS.
>
> The app is a Flask URL shortener. CI lints with flake8, runs pytest, builds the image, and curls the endpoints in a container smoke test. Then it stands up a kind cluster, provisions it with Terraform, deploys with Helm, and runs an end-to-end smoke test against the live app, all in GitHub Actions on every push.
>
> The Terraform shape maps onto AWS EKS: swap the kind provider for the EKS module, update a couple of variables, and the same Helm chart deploys unchanged. The app ships with a Prometheus /metrics endpoint and a Grafana dashboard.
>
> #cicd #terraform #kubernetes #flask

**Bluesky:**
> Flask URL shortener, full CI/CD: GitHub Actions (test → docker → smoke), Terraform on the kind provider, Helm deploy, end-to-end kind smoke test. Free and local, and the same Terraform maps to AWS EKS.
> github.com/mbongowo/DEVOPS

### DEVOPS/cicd-pipeline-petclinic — canonical Java pipeline
Status: unused

**Facts:** Spring PetClinic; multi-stage Docker build with cached dependency layer, JRE-only base, non-root runtime; GitHub Actions: mvnw verify → build → test → GHCR push; main-only gated production deploy with required reviewers; immutable commit-SHA image tags for deterministic rollback; ephemeral GITHUB_TOKEN, .env git-ignored.

**LinkedIn:**
> A canonical CI/CD pipeline for a real Java app: Spring PetClinic, packaged into a minimal container, tested on every push, published to GHCR, and gated for production.
>
> The Dockerfile is multi-stage: dependencies resolve and cache in the first stage, so a source-only change doesn't rebuild them; the second stage copies just the jar onto a JRE base, no build tools. Maven compiles and runs the full test suite, and if tests fail nothing is built. On a push to main the image goes to GHCR, then a deploy job waits for manual approval.
>
> Images tag by immutable commit SHA, so a rollback just redeploys a known-good hash. The app runs as a non-root user, .env is git-ignored, and CI uses the built-in GITHUB_TOKEN with no external PAT to rotate.
>
> #java #cicd #docker #devops

**Bluesky:**
> Spring PetClinic CI/CD: Maven test, multi-stage Docker (JRE-only, non-root), GHCR push, gated production deploy with required reviewers. Immutable SHA tags for rollback. No secrets in git.
> github.com/mbongowo/DEVOPS

### DEVOPS/terraform-azure-infra — free-tier IaC with guardrails
Status: unused

**Facts:** modular Terraform for Azure, free-tier only; input validation rejects any SKU beyond F1/B1; provisions resource group, VNet, subnet, NSG, Linux App Service Plan (F1); CI runs fmt -check, init -backend=false, validate (fully offline); optional Azure-authenticated plan on main when AZURE_PLAN_ENABLED=true; remote state documented but local by default.

**LinkedIn:**
> Most IaC walkthroughs assume you'll pay for the cloud. This Terraform provisions an Azure free-tier footprint and guards against accidentally creating anything billable.
>
> Input validation rejects any App Service SKU beyond F1 or B1, so a costly plan fails before it runs. The modules are clean: resource_group, network, and webapp, each independently testable, wired together at the root.
>
> CI runs entirely offline on every push and PR — fmt -check, init -backend=false, validate — so mistakes surface in a PR with no cloud credentials. An authenticated plan is optional on main behind a repo variable, and remote state is documented and one uncomment away.
>
> #terraform #iac #azure #devops

**Bluesky:**
> Modular Terraform for Azure, free-tier only: validation rejects non-F1/B1 SKUs, provisions VNet + NSG + F1 App Service. CI validates offline (fmt / init / validate). Remote state ready. Zero cost by default.
> github.com/mbongowo/DEVOPS

### DEVOPS/k8s-online-boutique-helm — one template, 11 services
Status: unused

**Facts:** Google Online Boutique (11 microservices + Redis); single generic Deployment/Service template driven by values.yaml; hardened defaults (non-root uid 1000, drop ALL caps, read-only rootfs, seccomp); per-service probe type (gRPC/HTTP/TCP/none); renders 25 resources; helm lint + kubeconform offline; kind dry-run smoke test in CI; add a service = one values entry.

**LinkedIn:**
> Helm charts bloat when every service is a copy-pasted Deployment and Service. This one isn't.
>
> Online Boutique (Google's 11-service demo plus Redis) packs into two generic templates. Each service is a map entry in values.yaml, and a template loop renders the rest: image, probe type, resources, environment. Adding a service is one entry, no template change.
>
> Hardening is defined once and applied to all 11 services: non-root uid 1000, all Linux capabilities dropped, read-only root filesystem. Probes are per-service (gRPC, HTTP, or TCP). Everything validates offline with helm lint and kubeconform, and CI dry-runs the install against a real kind API server.
>
> #helm #kubernetes #microservices #devops

**Bluesky:**
> Online Boutique Helm chart: 11 services rendered from one data-driven template, one values.yaml entry each. Hardened defaults (non-root, drop caps, read-only root). Validates offline + kind smoke test.
> github.com/mbongowo/DEVOPS

### DEVOPS/docker-compose-voting-app — a real e2e smoke test
Status: unused

**Facts:** five services (Flask vote, Redis queue, Python worker, Postgres, Node result); build contexts, healthchecks, depends_on service_healthy conditions, dual front/back networks (data layer internal-only), named volume; CI brings the stack up, casts votes over HTTP, and asserts end-to-end propagation; optional seed load generator; resource limits.

**LinkedIn:**
> Docker Compose isn't just for local development. This voting app uses the features you actually reach for in anger.
>
> Flask captures votes and pushes them to Redis; a Python worker pops the queue and upserts to Postgres; a Node page renders the live tally. Healthchecks on every service (Redis ping, Postgres pg_isready, HTTP /healthz) feed depends_on conditions, so the worker won't start until Redis and Postgres are healthy. Two networks isolate the data layer: only the vote and result services publish ports.
>
> The CI doesn't just lint the YAML. It brings the whole stack up, waits for healthy, casts votes through the real HTTP endpoint, and asserts they propagate end to end before tearing down.
>
> #docker #compose #microservices #devops

**Bluesky:**
> Docker Compose voting app: Flask/Redis/worker/Postgres/Node, dual networks, healthchecks, depends_on conditions, named volume. CI runs a live end-to-end smoke test (vote → redis → worker → db → result).
> github.com/mbongowo/DEVOPS

### DEVOPS/ansible-server-provisioning — idempotence, proven
Status: unused

**Facts:** roles common/docker/webserver/firewall; SSH hardening (no root login, no password auth), admin user, Docker apt repo, nginx templated site, ufw rules; Molecule tests converge + idempotence (zero changes on second run) + verify on a systemd container; ansible-lint, yamllint, syntax check in CI.

**LinkedIn:**
> Server provisioning belongs in Ansible roles, not shell scripts you copy between boxes.
>
> I wrote four roles: common (base OS hardening, an admin user, SSH locked down to keys only), docker (official repo plus the compose plugin), webserver (nginx with a templated site), and firewall (ufw with explicit SSH/HTTP/HTTPS rules). They compose into a tag-scoped site.yml.
>
> The testing is the point. Molecule stands up a systemd container, runs the roles, then runs them again and asserts zero changes the second time, which is the definition of idempotence. A verify step then checks the admin sudo access, Docker, nginx, and the served page. CI runs yamllint, ansible-lint, a syntax check, and the full Molecule suite.
>
> #ansible #infrastructure #linux #devops

**Bluesky:**
> Ansible server provisioning: 4 roles (common / docker / webserver / firewall). Molecule proves idempotence (zero changes on re-run) + verifies the result. yamllint, ansible-lint, syntax check in CI.
> github.com/mbongowo/DEVOPS

### DEVOPS/custom-github-action — a tested JS action
Status: unused

**Facts:** custom JavaScript GitHub Action (TypeScript source, ncc-bundled dist/index.js); computes next semantic version from base + release type (major/minor/patch/prerelease + preid); outputs previous/next version and tag; Jest covers all bump semantics; CI type-checks, tests, rebundles, guards against a stale bundle, and runs the action on itself.

**LinkedIn:**
> A GitHub Action can be plain JavaScript committed to your repo. I wrote one that computes the next semantic version.
>
> Input a current version and a release type (major, minor, patch, prerelease); output the next version and a git tag. The source is TypeScript, type-checked and bundled with ncc into dist/index.js, which GitHub runs directly, no Docker or registry.
>
> Jest covers the bump semantics: 1.4.2 + patch is 1.4.3, + major is 2.0.0, and a prerelease increments the rc counter. CI type-checks, tests, rebundles, fails if the committed bundle is stale, then runs the action on itself with uses: ./ and checks the outputs. It's a template for extracting repeated CI logic into a reusable step.
>
> #github #cicd #actions #typescript

**Bluesky:**
> Custom GitHub Action in TypeScript: computes the next semantic version (major/minor/patch/prerelease). Jest tests every bump, ncc-bundled, and CI runs the action on itself to test it. Zero dependencies.
> github.com/mbongowo/DEVOPS

### DEVOPS/gitops-argocd — App-of-Apps at scale
Status: unused

**Facts:** Argo CD App-of-Apps (one root.yaml cascades via directory.recurse); AppProject guardrails restrict source repos/namespaces and cluster-scoped kinds; ApplicationSet generates dev/prod from one template; Kustomize overlays; deploys the Online Boutique Helm chart + platform-config; manual bootstrap then Argo CD reconciles; CI validates overlays + Argo CD CRDs offline.

**LinkedIn:**
> Argo CD at scale is the App-of-Apps pattern: apply one root Application, and everything else reconciles from there.
>
> A single root.yaml uses directory.recurse to pick up every AppProject, Application, and ApplicationSet in the tree. An AppProject guardrails what the stack may deploy (which repos, namespaces, and resource kinds), and an ApplicationSet generates dev and prod from one template, so a new environment is one list entry. Each app points at a Kustomize overlay or Helm repo, and Argo CD keeps live state matching Git with automatic drift correction.
>
> CI validates it all offline: kustomize build each overlay, kubeconform the output, and validate the Argo CD CRDs, so mistakes surface in the PR.
>
> #gitops #argocd #kubernetes #devops

**Bluesky:**
> Argo CD App-of-Apps: a root.yaml bootstrap, AppProject guardrails, and an ApplicationSet generating per-environment apps from one template. Kustomize overlays, offline CRD validation. Git is the live state.
> github.com/mbongowo/DEVOPS

### DEVOPS/observability-stack — alert rules you can unit-test
Status: unused

**Facts:** Prometheus + Alertmanager + Grafana via docker-compose; node-exporter + cAdvisor; recording rules (5m request rate, error ratio); four alerting rules (TargetDown, HighErrorRate, HighNodeMemory, HighNodeCPU) unit-tested with promtool; provisioned Grafana dashboard + datasource; CI runs promtool, amtool check-config, docker compose config; one command to a full stack.

**LinkedIn:**
> This monitoring stack is fully transparent: Prometheus, Alertmanager, and Grafana orchestrated with docker-compose, every config testable.
>
> Prometheus scrapes node-exporter and cAdvisor and runs recording rules to pre-compute the 5-minute request rate and error ratio. Alertmanager routes alerts and inhibits noisy ones (no high-error spam when the target is already down). Grafana is provisioned as code, datasource and dashboard included.
>
> The rigor is in the tests: promtool feeds synthetic metrics into each alert and asserts it fires with the right labels — TargetDown after two minutes down, HighErrorRate on a 10% 5xx rate but not 5%. CI also runs amtool check-config and docker compose config, so everything passes before the stack comes up.
>
> #prometheus #monitoring #grafana #devops

**Bluesky:**
> Prometheus / Alertmanager / Grafana via docker-compose: node-exporter + cAdvisor, recording rules, and four alert rules unit-tested with promtool. amtool + compose validated in CI. One command to a full stack.
> github.com/mbongowo/DEVOPS
