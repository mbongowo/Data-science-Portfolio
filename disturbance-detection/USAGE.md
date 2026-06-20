# Usage

This guide walks through one full run: install the environment, edit the
configuration, build an NDVI time cube from a STAC catalogue, decompose each
pixel into trend and seasonality, detect the largest breakpoint, render date and
magnitude maps, and validate the detections against a documented event.

The pure-numpy core (`harmonic_decompose`, `detect_breakpoint`,
`spatial_agreement`) runs with nothing but numpy. The cube build needs the
geospatial stack and live access to a STAC API, so it is the only part that
cannot run offline.

## 1. Install

Two paths. `pixi` resolves the GDAL-backed geo wheels from conda-forge, which is
less fragile than pip for `rasterio`/`odc-stac`.

```bash
pixi install        # resolves conda-forge deps and writes pixi.lock
pixi run test       # pytest; the numpy core tests pass with no GIS install
```

pip fallback:

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

Target Python is 3.10-3.12. If you only want the decomposition and detection
functions, `pip install numpy` and `pip install -e .` are enough; the EO imports
stay dormant until you call `build_ndvi_cube`.

## 2. Configure `config/aoi.yaml`

Every field, grouped by block:

```yaml
aoi:
  name: "creek-fire-sierra-nf"     # label used in filenames and logs
  bbox: [-119.40, 37.10, -119.00, 37.45]  # [min_lon, min_lat, max_lon, max_lat], EPSG:4326
  crs: "EPSG:4326"                 # CRS of the bbox above

time:
  start: "2018-01-01"              # ISO start of the analysis record
  end: "2022-12-31"               # ISO end; span several years so the
                                   #   seasonal-trend model has cycles to fit
  resample_freq: "16D"            # pandas offset alias for the regular time grid

source:
  collection: "hls"               # "hls" (HLSS30/HLSL30) or "landsat-c2-l2"
  stac_url: "https://planetarycomputer.microsoft.com/api/stac/v1"
  resolution: 30.0                # output pixel size in metres
  index: "ndvi"                   # vegetation index (NDVI is what cube.py builds)

decompose:
  period_days: 365.25             # length of one seasonal cycle, units of the time axis
  n_harmonics: 2                  # sin/cos pairs: 1 = annual, 2 adds the
                                  #   semiannual asymmetry (green-up vs senescence)

detect:
  method: "cusum"                 # "cusum" (pure numpy) or "ruptures" (PELT, optional)
  min_segment: 5                  # minimum samples either side of a candidate break
  threshold: 1.0                  # normalised CUSUM score needed to flag a break
  magnitude_threshold: -0.1       # NDVI drop a detection must reach to count as a loss

validation:
  event_name: "Creek Fire 2020"
  event_date: "2020-09-04"        # reference date (MTBS ignition)
  window_days: 60                 # a detection agrees if within +/- this many days
  event_geojson: "data/raw/creek_fire_perimeter.geojson"  # event footprint polygon
  event_bbox: [-119.35, 37.15, -119.05, 37.40]            # fallback if no polygon
```

Pick `bbox`, `start`/`end`, and the `validation` block to match the event you
want to test. Keep the span at three or more years: the harmonic regression
needs several seasonal cycles before a single-year disturbance stands out from
the fitted seasonality. The `resample_freq` controls how dense the time axis is;
16-day composites suit HLS, while monthly (`"ME"`) suits a sparser Landsat-only
record.

## 3. Build the cube

The cube step follows the same cloud-native pattern as the eo-monitor project:
`odc.stac.load` over a STAC search, cloud/shadow masking from the scene QA band,
then NDVI, applied to a multi-year time axis rather than a single scene. Masked
pixels become `NaN`, never `0`, so a cloud gap is not read as a real low-NDVI
observation.

```bash
disturb --config config/aoi.yaml
```

or in Python:

```python
from disturb.cube import build_ndvi_cube

cube = build_ndvi_cube(
    bbox=[-119.40, 37.10, -119.00, 37.45],
    start="2018-01-01", end="2022-12-31",
    collection="hls", resolution=30.0, freq="16D",
)
# cube: xarray.DataArray, dims (time, y, x), Dask-backed, NaN where cloudy.
```

### Live STAC and Planetary Computer auth

`build_ndvi_cube` hits a live STAC API. With the default Planetary Computer URL,
asset hrefs are signed automatically through `planetary_computer.sign_inplace`
(wired into the catalogue search in `cube.py`). Anonymous access works for the
public HLS and Landsat C2 collections, but is rate-limited. For heavier use set a
subscription key:

```bash
export PC_SDK_SUBSCRIPTION_KEY=<your-key>     # Windows: setx PC_SDK_SUBSCRIPTION_KEY <key>
```

Get a key from the Planetary Computer Hub account page. For a different STAC
backend (for example a private catalogue or USGS Landsat via earthaccess), set
`source.stac_url` and supply that backend's credentials by its own mechanism;
the `sign_inplace` modifier is specific to Planetary Computer.

## 4. Decompose then detect

The cube is `(time, y, x)`. Apply the two pure-numpy functions per pixel. The
clearest way is `xarray.apply_ufunc` over the time dimension:

```python
import numpy as np
import xarray as xr
from disturb.decompose import harmonic_decompose
from disturb.detect import detect_breakpoint

t_days = (cube["time"].values - cube["time"].values[0]) / np.timedelta64(1, "D")

def _pixel(series):
    if np.isfinite(series).sum() < 6:        # too sparse to fit
        return np.nan, np.nan
    fit = harmonic_decompose(series, t=t_days, period=365.25, n_harmonics=2)
    bp = detect_breakpoint(fit.residual, times=cube["time"].values,
                           min_segment=5, threshold=1.0)
    if not bp.detected:
        return np.datetime64("NaT"), np.nan
    return bp.date, bp.magnitude

# Run _pixel over (y, x), consuming the time axis. For a first pass, loop the
# pixels; for the full AOI, wrap _pixel in xr.apply_ufunc with
# input_core_dims=[["time"]], vectorize=True, dask="parallelized".
```

Detection runs on the **residual** of the decomposition, so seasonality does not
masquerade as a break. `detect_breakpoint` returns the index, the calendar date
(when `times` is passed), and a signed magnitude: negative for an NDVI drop
(fire, clearing, drought), positive for a rise (regrowth). A single uncaught
cloud scores below a sustained step, so the CUSUM threshold separates one-off
outliers from real disturbances.

## 5. Date and magnitude maps

Collect the per-pixel `date` and `magnitude` into two 2-D arrays aligned to the
cube grid, then write them to `outputs/`:

- `outputs/disturbance_date.png` - date of the largest breakpoint per pixel.
- `outputs/disturbance_magnitude.png` - the NDVI drop at that breakpoint.

The README's result table points at exactly these two filenames. A `GeoTIFF`
alongside the PNG keeps the maps georeferenced; write it with
`rioxarray`'s `.rio.to_raster()` on a DataArray that carries the cube's CRS and
transform.

## 6. Validate against a documented event

Fetch the event footprint, rasterise it onto the cube grid, and compare:

```bash
python data/download.py --url <perimeter.geojson> --out data/raw/creek_fire_perimeter.geojson
```

```python
from disturb.validate import rasterize_event, spatial_agreement, summary

mask = rasterize_event("data/raw/creek_fire_perimeter.geojson", like=cube.isel(time=0))
result = spatial_agreement(
    detected_dates=date_map,           # 2-D datetime64, NaT where nothing fired
    detected_magnitude=magnitude_map,  # 2-D float, negative for a loss
    event_mask=mask,
    event_date="2020-09-04",
    window_days=60,
    magnitude_threshold=-0.1,
)
summary(result)
```

`spatial_agreement` reports the detection rate (event pixels flagged within the
window and past the magnitude threshold) and the false-alarm rate (background
pixels flagged in the same window). If you lack a polygon, build a rectangular
mask from `validation.event_bbox` instead of calling `rasterize_event`.

For the Creek Fire default, expect detections clustered inside the burn
perimeter within roughly two weeks of the 2020-09-04 ignition, a sharp negative
magnitude where the fire burned hottest, and few flags in the surrounding
unburned forest.

## Troubleshooting

**Sparse time series.** If a pixel has fewer finite samples than the model has
free parameters (`2 + 2 * n_harmonics`), `harmonic_decompose` raises. Guard with
a minimum-count check before the call (the `_pixel` example uses 6), drop
`n_harmonics` to 1, or widen `start`/`end` so more cycles are available. Sparse
pixels are common at AOI edges and over persistent cloud.

**Cloud gaps.** Gaps are `NaN`, not `0`, which both `harmonic_decompose` and
`detect_breakpoint` tolerate. If detections look noisy, the QA masking may be too
loose: check `_cloud_mask` in `cube.py` for the collection's bit layout, and
consider a longer `resample_freq` so each composite pools more clear looks. A
zero left in the series instead of a `NaN` manufactures phantom drops.

**CRS.** The `aoi.bbox` is EPSG:4326. The event polygon may arrive in a
different CRS; `rasterize_event` reprojects it to the cube's CRS with
`.to_crs(like.rio.crs)`, so the cube must carry a CRS for the rasterise to align.
If the mask lands offset from the imagery, confirm the cube's `rio.crs` and
`rio.transform()` are set and that the polygon reprojected cleanly.
