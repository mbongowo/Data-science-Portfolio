"""eo-explorer-app: a shippable interactive Earth-observation explorer.

A Streamlit + folium web app that lets a user draw an area of interest (AOI),
pick a date, pull live Sentinel-2 L2A imagery from the Earth Search STAC
catalogue, and render a spectral index (NDVI / NDWI / NDMI) on an interactive
map. The index functions are *reused* from the sibling ``eo-monitor`` package so
that the projects in this portfolio visibly compose.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
