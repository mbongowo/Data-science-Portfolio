# Social media post drafts

Ready-to-schedule posts promoting the portfolio and its live demos. Each entry
has a **LinkedIn** version (longer, professional) and a **short version** for
X/Bluesky. Hashtags are tuned for LinkedIn; X/Bluesky generally do better with
0–2 tags, so most are dropped there.

**Live demo links used below:**

- eo-explorer-app — https://data-science-portfolio-kpnqhpxmfzgwpgwbxkejql.streamlit.app/
- clinic-access dashboard — https://mbongowo-dat-spatial04-leafmap-dashboardappstreamlit-app-mclndk.streamlit.app/
- crop recommender — https://mbongowo-data-s-non-spatial04-ml-web-appappstreamlit-app-l5yxjk.streamlit.app/
- portfolio RAG — https://data-science-portfolio-fexfuaen4zdrzgpmu53nhs.streamlit.app/

## Scheduled in Metricool (brand 6363146, America/Chicago)

All 8 items were scheduled on 2026-06-23 to **LinkedIn + Facebook** (long copy)
and **Bluesky** (short copy) as separate posts, auto-publish on. YouTube was
excluded: it requires a video on every post, so the text/link content here does
not apply — see `README` of any demo project for screen-recording ideas if a
YouTube track is wanted later.

| When (Central) | Post | Networks |
|---|---|---|
| Tue Jun 23, 5:30 PM (immediate) | 1 — Portfolio anchor | LI + FB + Bluesky |
| Wed Jun 24, 9:00 AM | 2 — eo-explorer demo | LI + FB + Bluesky |
| Fri Jun 26, 9:00 AM | 3 — eo-monitor pipeline | LI + FB + Bluesky |
| Mon Jun 29, 9:00 AM | 4 — clinic dashboard | LI + FB + Bluesky |
| Wed Jul 1, 9:00 AM | 7 — engine bake-off | LI + FB + Bluesky |
| Fri Jul 3, 9:00 AM | 5 — crop recommender | LI + FB + Bluesky |
| Mon Jul 6, 9:00 AM | 8 — reproducibility | LI + FB + Bluesky |
| Wed Jul 8, 9:00 AM | 6 — portfolio RAG | LI + FB + Bluesky |

The "X/Bluesky" copy below is what was posted to Bluesky (kept under the
300-char limit); the "LinkedIn" copy was used for both LinkedIn and Facebook.

---

## Post 1 — Portfolio overview (anchor post)

**LinkedIn:**

> I've published my data science portfolio: 24 projects across a spatial / remote-sensing track and a big-data track, plus technique-replication builds.
>
> The rule I held myself to: every project's numerical core has a real known-answer test suite that passes in CI, and each ships a one-command demo that reproduces the numbers in its README. No screenshots standing in for results.
>
> Four projects are deployed as live apps you can click through today. Repo and live demos in the comments.
>
> #DataScience #GeospatialAnalytics #MachineLearning #RemoteSensing

**X/Bluesky:**

> Published my data science portfolio: 24 projects, spatial + big-data tracks, 4 live apps. Every project's core is tested in CI and ships a one-command reproducible demo. Links below 👇

---

## Post 2 — eo-explorer-app (flagship live demo)

**LinkedIn:**

> Draw an area on a map, pick a date and an index, and see live Sentinel-2 imagery rendered back to you. That's eo-explorer-app, the deployable app in my spatial track.
>
> The detail I'm proud of: the app imports its NDVI/index code directly from the eo-monitor pipeline rather than keeping a second copy. The app and the pipeline share one definition of the math, so they can't drift apart.
>
> Live demo: https://data-science-portfolio-kpnqhpxmfzgwpgwbxkejql.streamlit.app/
>
> #RemoteSensing #Sentinel2 #Geospatial #Python

**X/Bluesky:**

> Built a Streamlit app: draw an AOI, pick a date + index, see live Sentinel-2 on the map. It imports its index math straight from my eo-monitor pipeline, so app and pipeline never drift.
> ▶ https://data-science-portfolio-kpnqhpxmfzgwpgwbxkejql.streamlit.app/

---

## Post 3 — eo-monitor (cloud-native EO pipeline)

**LinkedIn:**

> One command, no manual downloads: eo-monitor pulls Sentinel-2 from a STAC catalogue over your area of interest, computes vegetation and moisture indices, scores anomalies against a baseline, and writes cloud-optimised GeoTIFFs.
>
> It's built on STAC, Dask, and COGs so it scales without a download folder full of scenes. Two other projects in the portfolio reuse its STAC-to-xarray cube pattern instead of reinventing it.
>
> #EarthObservation #STAC #Dask #GeospatialEngineering

**X/Bluesky:**

> eo-monitor: one command pulls Sentinel-2 from a STAC catalogue, computes veg/moisture indices, scores anomalies vs a baseline, writes COGs. STAC + Dask + cloud-optimised GeoTIFFs, no manual downloads.

---

## Post 4 — access-to-care / clinic dashboard (impact angle)

**LinkedIn:**

> Who is farthest from a clinic? I built a deployable dashboard that computes travel time from each populated place to the nearest health facility over a real road network, weighted by population — built around Cameroon.
>
> It's vector and network analysis turned into something a non-technical stakeholder can actually explore.
>
> Live: https://mbongowo-dat-spatial04-leafmap-dashboardappstreamlit-app-mclndk.streamlit.app/
>
> #GIS #PublicHealth #SpatialAnalysis #DataForGood

**X/Bluesky:**

> Who's farthest from a clinic? A dashboard computing population-weighted travel time to the nearest health facility over a real road network, built around Cameroon.
> ▶ https://mbongowo-dat-spatial04-leafmap-dashboardappstreamlit-app-mclndk.streamlit.app/

---

## Post 5 — crop recommender

**LinkedIn:**

> Soil and climate in, a ranked list of crops out. My crop-recommender takes local conditions and returns the crops best matched to them — deployed as a live app you can try right now.
>
> Live: https://mbongowo-data-s-non-spatial04-ml-web-appappstreamlit-app-l5yxjk.streamlit.app/
>
> #MachineLearning #Agriculture #Streamlit #AppliedML

**X/Bluesky:**

> Crop recommender: feed it soil + climate, get a ranked list of crops. Live app 👇
> ▶ https://mbongowo-data-s-non-spatial04-ml-web-appappstreamlit-app-l5yxjk.streamlit.app/

---

## Post 6 — portfolio RAG

**LinkedIn:**

> I built a RAG question-answering app over my own portfolio's documentation. Ask it about any project and it answers from the docs — free extractive mode by default, with an optional LLM path.
>
> A small, honest take on retrieval-augmented QA: no vector-store theatre, just a working answer engine over real documents.
>
> Live: https://data-science-portfolio-fexfuaen4zdrzgpmu53nhs.streamlit.app/
>
> #RAG #LLM #NLP #MachineLearning

**X/Bluesky:**

> Built a RAG Q&A app over my portfolio's docs — ask about any project, get answers from the source. Free extractive by default, optional LLM path.
> ▶ https://data-science-portfolio-fexfuaen4zdrzgpmu53nhs.streamlit.app/

---

## Post 7 — engine bake-off (big-data credibility)

**LinkedIn:**

> Which engine should run your analytical workload? I ran an honest bake-off on billions of NYC taxi rows: the same query across Spark, DuckDB, and a warehouse, benchmarked with real numbers rather than vibes.
>
> It's one of eight big-data projects in my portfolio, each scoped to a genuinely large public dataset with an architecture that runs locally or on a free cloud tier.
>
> #DataEngineering #Spark #DuckDB #BigData

**X/Bluesky:**

> Honest engine bake-off on billions of NYC taxi rows: same workload across Spark, DuckDB, and a warehouse, benchmarked. Numbers, not vibes.

---

## Post 8 — reproducibility (thought-leadership)

**LinkedIn:**

> A portfolio claim I see often: "achieved 94% accuracy." A question I rarely see answered: can I reproduce it?
>
> Across my 24 projects I committed to one standard — every numerical core has a known-answer test that runs in CI, and a one-command demo regenerates the numbers in the README from a seeded input. The heavy parts (GPU, Kafka, Terraform) are documented to run on your machine, not faked in a screenshot.
>
> Reproducibility isn't a feature you add at the end. It's the thing that makes the rest believable.
>
> #DataScience #Reproducibility #MLOps #SoftwareEngineering

**X/Bluesky:**

> "Achieved 94% accuracy" — but can you reproduce it? Across 24 portfolio projects I held one rule: every core has a known-answer test in CI + a one-command demo that regenerates the README numbers. Reproducibility first.
