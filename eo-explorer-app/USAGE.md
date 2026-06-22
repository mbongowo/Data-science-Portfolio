# EO Explorer usage guide

This guide covers installing the app, running it locally, working the user
interface, what the cache does, the messages you see when something is off, and
how to deploy to three hosts.

## Install

The app needs two things: its own runtime dependencies, and the `eo-monitor`
package, which supplies the spectral-index functions. `eo-monitor` is kept out
of the hard dependency list so the app installs and the tests run before that
package is built. You add it yourself in one of the forms below.

### pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # so `from app import ...` resolves
```

Then install `eo-monitor`. If you have a sibling checkout at `../eo-monitor`,
an editable install is the most convenient for development:

```bash
pip install -e ../eo-monitor
```

Without a local checkout, install it from git (the `eo-monitor` folder of this repo):

```bash
pip install "eo-monitor @ git+https://github.com/mbongowo/Data-science-Portfolio.git@main#subdirectory=eo-monitor"
```

The same git form is available as an extra:

```bash
pip install "eo-explorer-app[eo-monitor-git]"
```

### pixi

```bash
pixi install                  # resolves deps and writes pixi.lock
pixi run install-eo-monitor   # pip install -e ../eo-monitor
pixi run app                  # streamlit run app/main.py
```

To pull `eo-monitor` from git instead of a sibling path, uncomment the
`[pypi-dependencies]` line in `pixi.toml` before running `pixi install`.

### Verify the install

The pure-helper tests run without streamlit, folium, or the STAC stack, so
they are a fast check that the package imports correctly:

```bash
pytest -q
```

If `eo-monitor` is missing the tests still pass; the index functions fall back
to a local normalised-difference implementation that the tests confirm is
numerically identical. The app, however, refuses to compute an index without
the real package and tells you so.

## Run the offline demo

Before (or without) any of the satellite stack, you can run the pure core with
no network:

```bash
make demo          # or: python -m app.demo  /  pixi run demo
```

It validates a small synthetic AOI, synthesises one scene of bands with a
seeded generator, computes NDVI / NDWI / NDMI, prints the metrics, and writes
`outputs/summary.json` (plus `outputs/ndvi.png` when matplotlib and Pillow are
present). This is a synthetic scene, not satellite imagery, so it exercises the
maths and the plumbing rather than the data pipeline. The run is deterministic;
`tests/test_demo.py` pins the numbers. `notebooks/01_walkthrough.ipynb` walks
through the same demo with commentary.

## Run locally

```bash
streamlit run app/main.py
```

Streamlit opens a browser tab, usually at `http://localhost:8501`. The Makefile
target `make app` and the pixi task `pixi run app` wrap the same command.

If the sidebar shows a red banner saying `eo-monitor is not installed`, the app
runs and you can draw and validate an AOI, but the index step is disabled until
you install the package.

## Using the interface

The screen has a sidebar of controls and a map.

1. **Draw the area of interest.** Use the draw control on the map to place a
   rectangle or polygon. The bounding box of whatever you draw becomes the AOI.
   A FeatureCollection with several shapes is reduced to the box that contains
   all of them. Keep the area inside the size limit (see below) so the imagery
   loads quickly.

2. **Pick the date.** The date picker defaults to two weeks ago. The earliest
   selectable date is 2015-06-23, the start of the Sentinel-2 archive, and the
   latest is today. The app does not require imagery on the exact day: it
   searches a window of plus or minus ten days and keeps the least-cloudy scene
   it finds.

3. **Pick the index.** Three are offered, each reused from `eo-monitor`:

   - NDVI, vegetation greenness, drawn on a red-to-green ramp.
   - NDWI, open water, drawn on a blue ramp.
   - NDMI, vegetation moisture, drawn on a brown-to-green ramp.

   The sidebar shows a one-line description of the selected index.

4. **Set the maximum area.** A slider caps the AOI size between 100 and 5000
   square kilometres. The default is 2500. An AOI above the cap is rejected
   before any network call.

5. **Load.** Press *Load imagery & compute index*. The app validates the AOI,
   queries Earth Search, loads only the bands the index needs, computes the
   index, and adds a coloured layer to the map.

### Reading the layer and the legend

The coloured raster sits on top of the satellite basemap, clipped to your AOI.
A colour bar (the legend) maps colour to index value across a fixed range
chosen per index, so the same colour means the same value between runs. NDVI
runs roughly -0.2 to 0.9, NDWI -0.5 to 0.7, NDMI -0.4 to 0.6. Pixels with no
valid data, for example where a cloud mask removed reflectance, render
transparent and let the basemap show through.

Below the map, four numbers summarise the layer: the minimum, mean, and maximum
index value, and the fraction of pixels that carried valid data. A low valid
fraction usually means clouds covered much of the scene; try another date.

## Caching

Every scene search and band load is cached with Streamlit's `st.cache_data`,
keyed by the AOI bounding box, the date, and the index. The key is built by
`stac.cache_key`, which rounds the bounding box to six decimal places and
hashes the result, so the cache is stable across reruns and across processes.
Two consequences follow:

- Pressing *Load* again with the same area, date, and index returns instantly
  from the cache, with no network call.
- Nudging the AOI by less than about a tenth of a millimetre reuses the cached
  result, because the rounding treats those two boxes as the same request.

Changing any of the three inputs produces a new key and a fresh fetch. The
cache lives in the running Streamlit process; restarting the app clears it.

## Edge-case messages

The app states the single problem it found rather than failing silently. The
checks run in a fixed order and the first failure is the one you see.

- *Please draw an area of interest on the map first.* You pressed Load without
  drawing anything.
- *Could not read the drawn area: ...* The drawn shape had no usable
  coordinates.
- *Longitudes must be between -180 and 180 degrees.* / *Latitudes must be
  between -90 and 90 degrees.* A coordinate fell outside the valid range.
- *The area crosses the +/-180 deg date line ...* The AOI straddles the
  antimeridian, which the loader cannot handle. Draw on one side of it.
- *The drawn area is inverted (north edge below south edge).* The box is
  upside down.
- *The drawn area has no extent ...* You drew a point or a line, not a box.
- *Your area is about N km^2, which is larger than the M km^2 limit ...* The
  AOI exceeds the slider cap. Draw smaller or raise the slider.
- *No suitable Sentinel-2 scene was found for that area and date ...* The
  search window held no scene under the cloud threshold. Move the date.
- *Cannot compute the index because eo-monitor is not installed ...* Install
  the flagship package, then rerun.

## Deployment

Each host below serves the same `app/main.py`. The one detail they share is
that `eo-monitor` must be installed on the host, since it is not a hard
dependency. After a deploy succeeds, put the public URL at the top of
`README.md` and replace the screenshot placeholder.

### Streamlit Community Cloud

The repository is set up so this is mostly clicking through the deploy dialog.
`eo-monitor` is installed straight from the public repo, so there is no separate
step to publish it.

1. Push this repository to GitHub (already done for the portfolio).
2. At <https://share.streamlit.io>, sign in with GitHub and choose **Create app
   -> Deploy a public app from GitHub**.
3. Fill in the dialog:
   - Repository: `mbongowo/Data-science-Portfolio`
   - Branch: `main`
   - Main file path: `eo-explorer-app/app/main.py`
4. Open **Advanced settings** and set **Python version** to `3.12`. This matters:
   `eo-monitor` requires Python below 3.13, so the default may fail to install.
5. Click **Deploy**. Cloud finds `eo-explorer-app/app/requirements.txt` (it
   searches the entrypoint directory first), installs the dependencies including
   `eo-monitor` from git, and reads `.streamlit/config.toml`. The first build is
   slow because of the geospatial wheels; later builds reuse the cache.
6. When the app is live, copy the URL into the top of `README.md` and replace the
   screenshot placeholder with a real screenshot.

If the build fails while compiling a geospatial package, add a file named
`packages.txt` at the **repository root** (Cloud only reads it there) with:

```
gdal-bin
libgdal-dev
```

then redeploy. The pip wheels for rasterio and pyproj usually make this
unnecessary.

### Hugging Face Spaces

1. Create a new Space and choose the *Streamlit* SDK.
2. Push this repository into the Space (the Space is a git remote). Make sure
   the `eo-monitor` git line in `requirements.txt` is uncommented, as above.
3. Add `app_file: app/main.py` to the Space's `README.md` front matter so the
   runner launches the right entry point:

   ```yaml
   ---
   title: EO Explorer
   sdk: streamlit
   app_file: app/main.py
   ---
   ```

4. The Space builds from `requirements.txt` and starts the app. Watch the build
   log for the GDAL packages on the first run.

### Fly.io

The repository ships a `Dockerfile` that installs the system GDAL libraries,
the Python dependencies, and `eo-monitor` from git, then runs Streamlit on port
8501.

1. Install flyctl and sign in: `fly auth login`.
2. From the repository root, run `fly launch`. Accept the detected Dockerfile
   and decline the offer to deploy immediately so you can check the generated
   `fly.toml`.
3. Set the internal port to 8501 in `fly.toml` so Fly's proxy forwards to the
   Streamlit server:

   ```toml
   [http_service]
     internal_port = 8501
     force_https = true
   ```

4. Deploy with `fly deploy`. The build runs the Dockerfile; the `HEALTHCHECK`
   in the image polls Streamlit's `/_stcore/health` endpoint.
5. Open the app with `fly open`.

To build and run the same image locally first:

```bash
docker build -t eo-explorer-app .
docker run -p 8501:8501 eo-explorer-app
# open http://localhost:8501
```
