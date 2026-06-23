# EO Monitor — vegetation stress and anomaly detection from Sentinel-2

EO Monitor flags vegetation stress and anomalies in satellite imagery. It pulls
Sentinel-2 scenes through odc-stac and xarray, computes spectral indices such as
NDVI, NDWI and NDMI, and compares each pixel against a seasonal baseline. Pixels
whose index falls far below the baseline are flagged as anomalies using a
z-score test, so drought or crop stress shows up as a map of anomalous pixels.

The project demonstrates the pipeline on the 2023 Corn Belt flash drought, where
the NDVI anomaly map tracks the documented stress. The core is config-driven and
deployable, with rasterio for raster I/O. It is the index library that the EO
Explorer app reuses for its on-the-fly index rendering.
