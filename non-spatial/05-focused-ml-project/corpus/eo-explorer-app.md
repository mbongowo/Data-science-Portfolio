# EO Explorer App — interactive Sentinel-2 spectral index explorer

EO Explorer is an interactive Streamlit web app for exploring satellite imagery.
The user draws an area of interest on a folium map, picks a date and one of 34
spectral indices spanning vegetation, water, soil, built-up, snow and fire
categories, and the app renders live Sentinel-2 L2A imagery for that index over
the area. Results can be downloaded as a georeferenced GeoTIFF.

The app fetches scenes through a STAC catalogue and reuses the spectral index
functions from EO Monitor, so the same index maths powers both the batch monitor
and the interactive explorer. It is a deployable, no-Earth-Engine browser tool.
