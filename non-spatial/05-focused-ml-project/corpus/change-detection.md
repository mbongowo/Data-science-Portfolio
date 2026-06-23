# Change Detection — Otsu SAR flood mapping

Change Detection maps floods from Sentinel-1 SAR radar imagery. Radar sees
through cloud, so it works during storms. The method converts backscatter to a
dB scale and applies automatic Otsu thresholding to separate water from land in
before and after images, then classifies each pixel as flooded, permanent water
or receded, and reports the flooded area in hectares.

The case study is Cameroon's Logone floodplain, with results validated against
UN OCHA flood reporting. The core is pure numpy with an optional MNDWI water
index fallback. A separate Earth Engine time-series project handles optical
NDVI/NBR bitemporal change instead.
