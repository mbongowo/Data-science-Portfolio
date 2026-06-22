# How to run each project

This is the short, do-this version. Each project also has its own `README.md` and
`USAGE.md` with more detail, but you can run any of them with the steps here.

## Why pixi

Five of the six projects need GDAL and other geospatial C libraries. These are
painful to install with `pip` and easy with `pixi`, which installs them from
conda-forge. Install pixi once and every project becomes a two-line setup.

Install pixi:

- macOS / Linux: `curl -fsSL https://pixi.sh/install.sh | bash`
- Windows (PowerShell): `iwr -useb https://pixi.sh/install.ps1 | iex`

Close and reopen your terminal afterwards so `pixi` is on your PATH.

The pattern for every project is the same:

```bash
cd <project>      # e.g. cd spatial/eo-monitor
pixi install      # first time only: builds the environment and writes pixi.lock
pixi run <task>   # run something (tasks are listed per project below)
```

`pixi run test` works in every project and needs no network or API keys, so it
is the quickest way to confirm a project is set up correctly.

If you only want to read the code and run the unit tests without pixi, a plain
`pip install` in a virtual environment is enough for the tests; the full data
pipelines need the geospatial stack, which is why pixi is the recommended path.

---

## 1. eo-monitor — Sentinel-2 anomaly pipeline

No account or API key needed; it reads Sentinel-2 from the open Earth Search
catalogue.

```bash
cd spatial/eo-monitor
pixi install
pixi run run
```

`pixi run run` is `eo-monitor run --config config/corn_belt.yaml`. It searches for
imagery, computes NDVI/NDWI/NDMI and an anomaly against a baseline, and writes
Cloud-Optimised GeoTIFFs plus a preview PNG into `outputs/`. To change the area
or dates, edit `config/corn_belt.yaml`.

## 2. access-to-care — travel time to clinics

Needs the input data first. Most sources download without a key; the health
facility points come from Healthsites.io, which needs a free key.

```bash
cd spatial/access-to-care
pixi install

# Optional but recommended (free key from https://healthsites.io):
export HEALTHSITES_API_KEY=your_key          # Windows PowerShell: $env:HEALTHSITES_API_KEY="your_key"

pixi run download    # fetches OSM, facilities, population, and boundaries into data/raw
pixi run access      # computes travel time and the population-weighted equity stats
```

Outputs (an access-time map and a summary table) land in `outputs/`.

## 3. spatial-hotspots — cluster and hotspot statistics

Needs a free USDA NASS key for the crop-yield data.

```bash
cd spatial/spatial-hotspots
pixi install

# Free key from https://quickstats.nass.usda.gov/api:
export NASS_API_KEY=your_key                 # Windows PowerShell: $env:NASS_API_KEY="your_key"

pixi run download    # writes data/raw/iowa_corn_yield_2023.gpkg
pixi run analyze     # Moran's I, LISA, and Getis-Ord Gi*, with maps in outputs/
```

## 4. geoai-segmentation — segmentation model

This project trains a U-Net to segment building footprints from imagery. The full
run wants a GPU and a real labelled dataset, but you can exercise the whole
pipeline on synthetic data and CPU first.

```bash
cd spatial/geoai-segmentation
pixi install
```

### 4a. Quick wiring check (no GPU, no real data)

```bash
pixi run prepare-synthetic   # writes a tiny synthetic dataset to data/spacenet
pixi run smoke               # one training step on CPU, to prove the wiring works
```

If `pixi run smoke` finishes without error, training, the data module, the model,
and the loss are all wired correctly. The next steps swap in real data.

### 4b. Get a real dataset

The data module expects matching image and mask GeoTIFFs:

```
data/spacenet/images/<name>.tif
data/spacenet/masks/<name>.tif      # same <name>, single-band 0/1 mask
```

Two ways to get there:

- **SpaceNet building footprints** (AWS Open Data, no credentials, needs the AWS
  CLI installed):

  ```bash
  aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/ ./data/raw/ \
      --recursive --no-sign-request
  ```

  SpaceNet ships images with vector (GeoJSON) labels, so the polygons need to be
  rasterised into single-band masks. The included helper does that in one command,
  pairing each image with its GeoJSON, burning the footprints onto the image grid,
  and writing the `images/` + `masks/` layout above:

  ```bash
  pixi run python scripts/rasterize_spacenet.py \
      --images data/raw/RGB-PanSharpen \
      --labels data/raw/geojson/buildings \
      --out data/spacenet
  ```

  It pairs files on the shared `img<N>` token by default (override with
  `--key-regex`), reprojects labels to each image's CRS, and accepts `--all-touched`,
  `--link` (hardlink images instead of copying), and `--limit N` for a quick trial.
- **torchgeo benchmark loaders.** Run `python scripts/prepare_data.py --torchgeo`
  to print a ready-made snippet (for example `SpaceNet1(root="data/raw",
  download=True)`).

Or point the config at any folder that already follows the layout, with
`data.data_dir=...` (see below).

### 4c. Train

```bash
pixi run train
```

`pixi run train` is `python -m geoseg.train`. It composes `conf/config.yaml`
(`model: unet` + `data: spacenet`), seeds everything, trains, and logs the
resolved config and the current git SHA to MLflow so a run traces back to exact
code and settings.

Every setting is a Hydra override on the command line, so you rarely edit the YAML
directly:

```bash
pixi run python -m geoseg.train \
  data.data_dir=./data/spacenet data.batch_size=16 data.num_workers=8 \
  trainer.max_epochs=50 trainer.accelerator=gpu trainer.devices=1 \
  model.encoder_name=resnet50 model.lr=0.0005
```

- Force CPU with `trainer.accelerator=cpu`; the default `auto` uses a GPU if one
  is present.
- For multispectral input, set `model.in_channels` to your band count and update
  `data.band_means` / `data.band_stds`.
- For more than two classes, set `model.classes`.
- Sweep a hyperparameter with Hydra multirun: `... -m model.lr=0.001,0.0005`.

By default Hydra creates a fresh `outputs/<timestamp>/` directory per run and
writes the checkpoint, logs, and `mlruns` there (the path is printed when the run
starts). To keep everything in the project root instead, add `hydra.run.dir=.`:

```bash
pixi run python -m geoseg.train hydra.run.dir=.
```

### 4d. Find the trained checkpoint

Lightning saves a `.ckpt` under the run directory:

- macOS / Linux: `find outputs -name "*.ckpt"`
- Windows PowerShell: `Get-ChildItem -Recurse -Filter *.ckpt outputs`

### 4e. Evaluate on the held-out test split

This writes `metrics.json` (IoU and F1) and an image / ground-truth / prediction
panel PNG:

```bash
pixi run python -c "from geoseg.evaluate import evaluate; print(evaluate('PATH/TO.ckpt', 'data/spacenet', 'outputs/eval'))"
```

### 4f. Predict on a new tile

Writes a georeferenced GeoTIFF mask whose CRS and transform are copied from the
input:

```bash
pixi run python -m geoseg.infer run \
  --checkpoint PATH/TO.ckpt --input tile.tif --output mask.tif
```

(Installed console scripts `geoseg-train` and `geoseg-infer` do the same thing.)

### 4g. Inspect runs in MLflow

```bash
pixi run mlflow ui --backend-store-uri ./mlruns   # open http://localhost:5000
```

Point `--backend-store-uri` at wherever `mlruns` landed (project root if you used
`hydra.run.dir=.`, otherwise inside the timestamped run directory).

Because the seed, the resolved config, and the git SHA are all recorded, the same
seed and config reproduce the same metrics. `MODEL_CARD.md` documents the data,
metrics, and limitations.

## 5. disturbance-detection — change over time

Reads from Microsoft Planetary Computer, which works anonymously (a subscription
key only raises rate limits, and is not required).

```bash
cd spatial/disturbance-detection
pixi install
pixi run run
```

`pixi run run` is `disturb --config config/aoi.yaml`. It builds an NDVI time
series, fits a seasonal model, detects the largest breakpoint per pixel, and
writes date-of-change and magnitude maps to `outputs/`. Edit `config/aoi.yaml` to
change the area, dates, or detection settings.

## 6. eo-explorer-app — the interactive web app

This one runs in your browser. It reuses the index code from `eo-monitor`, so
install that first.

```bash
cd spatial/eo-explorer-app
pixi install
pixi run install-eo-monitor    # installs the sibling eo-monitor package
pixi run app                   # opens the app at http://localhost:8501
```

In the app: draw a small box over land with the rectangle tool, pick a recent
date and an index, then click **Load imagery & compute index**. To put it online
instead, see the deployment steps in `spatial/eo-explorer-app/USAGE.md`.

---

## Running the tests for everything

From inside any project folder:

```bash
pixi run test
```

This runs that project's unit tests. They use small built-in fixtures, so they
need no network access and no API keys. The same tests run automatically in CI on
every push.
