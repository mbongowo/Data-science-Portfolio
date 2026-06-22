# Profile README

Drop the section below the `---` into a repository named after your GitHub
account (`mbongowo/mbongowo`), in its `README.md`. GitHub shows that file at the
top of your profile page. (This repo now has it pushed there already; edit and
re-push when the portfolio changes.) After pushing, pin the `Data-science-Portfolio`
repository on your profile (Customize your pins) so it shows first.

---

### Hi, I'm Joseph Mbuh 👋

I work in data science across two areas: **spatial / remote sensing** (Python
geospatial pipelines, GeoAI, reproducible Earth-observation analysis) and
**large-scale "big data"** analytics (streaming, distributed processing, the
modern data stack). I care more about whether someone else can reproduce a result
from a clean clone than about how the notebook looked the day I wrote it.

**Portfolio:** [Data-science-Portfolio](https://github.com/mbongowo/Data-science-Portfolio)
— 14 projects in two tracks, each with its own environment, hand-derived
known-answer tests, CI, and docs. Every project ships a one-command reproducible
demo.

**🛰️ Live app — [EO Explorer](https://data-science-portfolio-kpnqhpxmfzgwpgwbxkejql.streamlit.app/):**
draw an area, pick a date and a spectral index, and see live Sentinel-2 rendered
on a map (the index maths is reused from the `eo-monitor` package).

[![EO Explorer](https://raw.githubusercontent.com/mbongowo/Data-science-Portfolio/main/spatial/eo-explorer-app/docs/screenshot.png)](https://data-science-portfolio-kpnqhpxmfzgwpgwbxkejql.streamlit.app/)

**Spatial track**
- **eo-monitor** — Sentinel-2 from STAC to spectral indices, anomaly maps, and cloud-optimised GeoTIFFs, one command, no manual downloads.
- **access-to-care** — travel time to the nearest clinic over a road network, weighted by population, built around Cameroon.
- **spatial-hotspots** — spatial autocorrelation and cluster maps (Moran's I, LISA, Getis-Ord Gi\*), with the interpretation and the limits written out.
- **geoai-segmentation** — a segmentation model set up so a reported metric reproduces from the committed seed and config, with a model card.
- **disturbance-detection** — NDVI time series, harmonic decomposition, and breakpoint detection of when and where the land changed.
- **eo-explorer-app** — the deployed interactive web app above, reusing the index code from `eo-monitor`.

**Big-data track**
- **clickstream-pipeline** (Kafka + Spark streaming), **log-anomaly** (Spark + anomaly detection on labelled logs), **als-recommender** (Spark MLlib), **sentiment-scale** (Spark + NLP), **tlc-analytics** (a Spark/DuckDB/warehouse engine bake-off), **dbt-modern-stack** (dbt + orchestration), **crypto-backtest** (a no-look-ahead backtest), **graph-analysis** (PageRank / communities / triangles at scale).

Each project's pure-numerical core is unit-tested and runs in CI (~800 known-answer
tests across the repo); the full data and compute pipelines are documented per
project.
