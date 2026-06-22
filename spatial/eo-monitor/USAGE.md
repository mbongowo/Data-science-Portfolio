# Using eo-monitor

Step-by-step: install, configure, run, interpret output. The troubleshooting
section at the end covers the most common failure modes (STAC search,
authentication, CRS).

## 1. Install

You need a Python between 3.10 and 3.12. The geospatial stack (GDAL, rasterio,
odc-stac) does not build cleanly on 3.13+, so do not use a newer interpreter.

### Option A: pixi (recommended)

[pixi](https://pixi.sh) resolves the whole stack from conda-forge, so you do not
have to fight GDAL wheels. From the repo root:

```bash
pixi install                                   # creates the env, writes pixi.lock
pixi run eo-monitor run --config config/corn_belt.yaml
```

`pixi.lock` is generated on your machine. It is not committed because the pinned
builds are platform-specific.

### Option B: pip

If you already have a working GDAL/rasterio on your system (or platform wheels
are available), install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
eo-monitor run --config config/corn_belt.yaml
```

The `requirements.txt` mirrors the runtime dependencies if you prefer to install
those directly.

### Check the install

```bash
eo-monitor --version
eo-monitor --help
```

These work without the heavy geo libraries because the CLI defers those imports
until a run actually starts. If `--help` works but a run fails on an import, the
problem is a missing geo dependency, not the package itself.

### Quick offline demo (no geo stack)

Before configuring a real run you can exercise the index and anomaly core with a
small seeded synthetic cube. It needs only numpy and finishes in under a second:

```bash
eo-monitor demo                 # or: pixi run demo / make demo
eo-monitor demo --seed 0 -o outputs
```

It synthesises baseline and target reflectance stacks, plants a vegetation-loss
patch in the target, runs the real NDVI/NDWI/NDMI and z-score anomaly code, and
prints a metrics JSON (mean NDVI, anomaly pixel count, max |z|, and the fraction
of the planted patch recovered by the |z| > 2 mask). It writes `summary.json`
plus the NDVI and z-score maps as `.npy` to the output directory. This is a
synthetic sanity check, not a real observation.

## 2. Configure

Everything the pipeline does comes from one YAML file. The shipped example is
`config/corn_belt.yaml` (2023 Corn Belt flash drought over eastern Nebraska).
Copy it and edit the copy for your own area and dates.

Every field:

### `aoi` (area of interest)

Set exactly one of `bbox` or `vector_path`. Setting both, or neither, is a config
error.

- `bbox`: `[lon_min, lat_min, lon_max, lat_max]` in EPSG:4326 (degrees). The
  validator checks `lon_min < lon_max`, `lat_min < lat_max`, and that the values
  are inside the valid lon/lat range. Keep the box small. A field-scale AOI keeps
  the download and compute cheap and the run reproducible.
- `vector_path`: path to a GeoJSON/GPKG/SHP file. Its bounds are read with
  geopandas, reprojected to EPSG:4326, and used as the search box. The geopandas
  import only happens on this path, so a bbox AOI does not need it.

### `date_range` (target window)

The period you are asking about, usually a stress event.

- `start`, `end`: ISO dates (`YYYY-MM-DD`), inclusive. `start` must be on or
  before `end`. Items in this window are composited (per-pixel median) into one
  map per index.

### `baseline` (climatological reference)

The "normal" the target is compared against. Make it span several years so the
per-pixel mean and standard deviation are stable.

- `start`, `end`: same rules as `date_range`. Use multiple years.
- `months`: list of month numbers (1-12) that contribute to the baseline
  statistics. Restricting to the target season (for example `[7, 8]`) compares
  like with like, so July 2023 is measured against past Julys and Augusts rather
  than against winter scenes. If omitted, all months are used.

### Search and grid settings

- `cloud_cover_max` (default 20): drop scenes whose `eo:cloud_cover` is at or
  above this percentage. Raising it admits more scenes but more cloud
  contamination; lowering it can leave you with too few items.
- `max_items` (default 60): hard cap on the number of STAC items pulled per
  window. A guard against an over-broad search. Increase it if your AOI or date
  range is large and you are missing scenes.
- `resolution` (default 20): output pixel size in metres. 10 m is available for
  some bands, 20 m matches SWIR (B11) used by NDMI.
- `crs` (default `EPSG:32614`): the projected CRS the cube is reprojected to.
  Set the UTM zone that covers your AOI. EPSG:32614 is UTM 14N (eastern
  Nebraska). The wrong zone still runs but distorts distances and pixel shapes.
- `groupby` (default `solar_day`): how odc-stac groups items before loading.
  `solar_day` mosaics scenes captured on the same day, which removes
  tile-boundary duplicates.

### `indices`

A list of one or more index names (case-insensitive; normalised to upper case).
Each index produces its own anomaly and composite output. The full catalogue,
grouped by category:

| Category | Indices |
|----------|---------|
| Vegetation | NDVI, EVI, EVI2, SAVI, MSAVI, GNDVI, ARVI, NDRE, VARI, RVI, DVI, CIgreen, CIrededge, MCARI, TCARI, LAI |
| Water & moisture | NDWI, MNDWI, NDMI, AWEI, NDII |
| Soil & geology | BSI, SI, IronOxide, ClayMinerals, FerrousMinerals |
| Built-up / urban | NDBI, UI, IBI |
| Snow / ice | NDSI, NDGI |
| Fire / burn | NBR, NBR2, BAI |

The classic three (`NDVI`, `NDWI`, `NDMI`) remain the defaults in the shipped
config. Selected formulae:

| Index | Formula | Bands (S2 L2A) | Sensitive to |
|-------|---------|----------------|--------------|
| NDVI  | (NIR − Red) / (NIR + Red)          | B08, B04 | green biomass / vigour |
| NDWI  | (Green − NIR) / (Green + NIR)      | B03, B08 | open water / wetness |
| NDMI  | (NIR − SWIR) / (NIR + SWIR)        | B08, B11 | canopy moisture |
| SAVI  | (1+L)(NIR − Red)/(NIR + Red + L)   | B08, B04 | vegetation, soil-corrected (L=0.5) |
| EVI   | 2.5(NIR−Red)/(NIR+6·Red−7.5·Blue+1)| B08, B04, B02 | dense canopy, aerosol-corrected |
| MNDWI | (Green − SWIR1) / (Green + SWIR1)  | B03, B11 | open water (built-up suppressed) |
| BSI   | ((SWIR1+Red)−(NIR+Blue))/(sum)     | B11, B04, B08, B02 | bare soil |
| NDBI  | (SWIR1 − NIR) / (SWIR1 + NIR)      | B11, B08 | built-up / impervious |
| NDSI  | (Green − SWIR1) / (Green + SWIR1)  | B03, B11 | snow / ice |
| NBR   | (NIR − SWIR2) / (NIR + SWIR2)      | B08, B12 | burn severity |

**Reflectance note.** Indices with additive constants (EVI, SAVI, MSAVI, AWEI,
BAI, LAI) need surface reflectance in `[0, 1]`; the pipeline's reflectance cube
already supplies that. Normalised-difference and ratio indices are scale-invariant.

**Caveats.** `NDII` is identical to `NDMI`; `NDSI` shares the `MNDWI` formula;
`SI` is the `sqrt(Green*Red)` salinity form; `NDGI` is the green/red glacier
variant; `LAI` is an approximate empirical relation from EVI.

**Not supported.** `EBBI` (needs a thermal band Sentinel-2 lacks). `dNBR` is not
a single-scene index — it is the differenced NBR between two dates, which this
pipeline expresses through its **anomaly** workflow (target window vs baseline)
rather than as an entry in this `indices` list.

### `stac`

- `url` (default `https://earth-search.aws.element84.com/v1`): the STAC API
  endpoint.
- `collection` (default `sentinel-2-l2a`): the collection ID at that endpoint.
  If you point `url` at Microsoft Planetary Computer instead, the collection ID
  is `sentinel-2-l2a` there too, but the assets need signing (see
  Troubleshooting).

### `output`

- `dir` (default `outputs`): where results are written. Created if missing.
- `write_quicklook` (default `true`): also write a PNG per anomaly.

A malformed config raises a `pydantic.ValidationError` with a message naming the
field, so read the first line of the error.

## 3. Run

```bash
eo-monitor run --config config/corn_belt.yaml
```

Flags:

- `--config`, `-c`: path to the YAML file (required).
- `--output-dir`, `-o`: override `output.dir` from the config.
- `--max-items`: override `max_items` for a quick, smaller test run.
- `--verbose`, `-v`: DEBUG logging.

Example of a fast smoke test that writes elsewhere:

```bash
eo-monitor run -c config/corn_belt.yaml --max-items 20 -o /tmp/run1 -v
```

The run logs five stages: STAC search (target, then baseline), cube build, index
computation, anomaly, and export. Each requested index is processed in turn.

There is also a `demo` subcommand (`eo-monitor demo --seed N -o DIR`) that runs
the offline synthetic example described under "Quick offline demo" above; it
needs no config and no geo stack.

## 4. Outputs

For the default config (all three indices) you get, under `outputs/`:

```
outputs/
├── ndvi_anomaly.tif    ndwi_anomaly.tif    ndmi_anomaly.tif      # COGs (z-score)
├── ndvi_composite.tif  ndwi_composite.tif  ndmi_composite.tif    # target medians
└── ndvi_anomaly.png    ndwi_anomaly.png    ndmi_anomaly.png       # quicklooks
```

- `*_anomaly.tif`: the per-pixel z-score, `(target_value - baseline_mean) /
  baseline_std`. A Cloud-Optimised GeoTIFF with overviews, NaN nodata.
- `*_composite.tif`: the median of the target window for that index, for
  reference (the raw value behind the anomaly).
- `*_anomaly.png`: the quicklook, color-scaled from −3 to +3 with the `RdYlGn`
  colormap.

The `outputs/` directory is git-ignored.

## 5. Reading the anomaly map

The value at each pixel is a z-score: how many baseline standard deviations the
target sits above or below the baseline mean. A z near 0 means the target year
matches the historical normal for that pixel.

- A negative z means the target is below normal. For NDVI that is less green
  biomass; for NDMI, drier canopy. In a drought map these are the stressed pixels.
- A positive z means higher than normal (greener or wetter).
- |z| ≥ 2 corresponds roughly to the 2.5 %/97.5 % tail of a normal baseline and
  is a reasonable threshold for "notably abnormal."

In the default quicklook (`RdYlGn`, −3 to +3) red is negative (stressed), green
is positive. NaN pixels are masked: either clouds removed by the SCL mask, or
pixels where the baseline standard deviation was 0 so a z-score is undefined.

The composite GeoTIFF tells you the absolute index level. Use it together with
the anomaly: a low NDVI that is also a low anomaly is genuinely degraded ground,
whereas a low NDVI that is a normal anomaly is just naturally sparse vegetation.

## 6. Troubleshooting

### "No STAC items matched"

The search returned nothing after filtering. In order of likelihood:

- `cloud_cover_max` is too low for the season. Raise it (for example to 40) and
  retry.
- The date range is too short. Widen `date_range` or the `baseline` window.
- The AOI is tiny or in the wrong place. Confirm the bbox is
  `[lon_min, lat_min, lon_max, lat_max]` in that order and in degrees. A swapped
  lon/lat or a sign error puts the box in the ocean.
- The baseline `months` filter removed everything. Check the months overlap the
  baseline date range.

### Connection or timeout errors during search

The STAC endpoint may be down or rate-limiting. Confirm the `stac.url` is
reachable and retry. Earth Search is anonymous, so a 401/403 here points at a
proxy or network policy, not credentials.

### 403 / unsigned-asset errors when loading the cube

Earth Search on AWS serves Sentinel-2 L2A from a public bucket and needs no
credentials. If you switched `stac.url` to Microsoft Planetary Computer, its
asset URLs are time-limited and must be signed with `planetary-computer.sign`
before loading. The default config uses Earth Search precisely to avoid this. If
you need Planetary Computer, sign the items between search and cube load.

### CRS / projection problems

- Output looks stretched or pixels are not square: `crs` is the wrong UTM zone
  for your AOI. Pick the zone that contains the AOI centre. UTM zone number is
  `floor((lon + 180) / 6) + 1`; northern hemisphere zones are EPSG `326NN`.
- The bbox itself is always EPSG:4326 (degrees), regardless of `crs`. Do not put
  projected metres in `bbox`.
- A vector AOI in another CRS is fine; it is reprojected to 4326 for the search
  automatically.

### Empty or all-NaN output

Every pixel is NaN when the cloud mask removed all observations, or when the
baseline had no variability. Loosen `cloud_cover_max`, widen the windows, or
check that the baseline spans enough distinct scenes to give a non-zero standard
deviation.

### Import errors on run but `--help` works

A geo dependency is missing or the interpreter is too new. Confirm Python is
3.10-3.12 and reinstall (`pixi install`, or `pip install -e ".[dev]"`). The pure
math in `indices.py` and `anomaly.py` needs only numpy and is exercised by the
offline tests; everything else needs the full stack.

### Running the tests

```bash
pytest                 # or: pixi run pytest
```

The `indices.py` and `anomaly.py` tests use hand-checked values and run offline
with only numpy. The config tests read `config/corn_belt.yaml`.
