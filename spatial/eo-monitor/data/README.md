# data/

This directory is **git-ignored** (see `.gitignore`). There are no manual
downloads: imagery is fetched on demand from a public STAC catalogue when you
run the pipeline.

## Where the data comes from

`eo-monitor` discovers Sentinel-2 L2A scenes from **Earth Search**
(`https://earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a`) and
streams only the bands and AOI window it needs via `odc-stac` (HTTP range reads
against the Cloud-Optimised GeoTIFFs on AWS Open Data). Nothing is staged here
unless you choose to.

## Download mechanism

Running the pipeline performs the fetch automatically:

```bash
pixi run eo-monitor run --config config/corn_belt.yaml
```

To use a **local vector AOI** instead of a bbox, place a file here and point the
config at it:

```yaml
aoi:
  vector_path: data/aoi/nebraska_field.geojson
```

Generated rasters (COGs, quicklooks) are written to `outputs/`, which is also
git-ignored.
