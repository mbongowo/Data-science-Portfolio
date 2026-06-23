# Disturbance Detection — harmonic + CUSUM breakpoints in NDVI time series

Disturbance Detection finds fires and deforestation in multi-year vegetation
time series. For each pixel it fits a harmonic (seasonal) model to the NDVI
history, then runs a CUSUM cumulative-sum test on the residuals to detect the
breakpoint where the series departs from its seasonal norm. The output is a map
of disturbance dates and magnitudes from Sentinel-2 and HLS imagery.

The pure-numpy core is validated against documented events such as the Creek
Fire, recovering the disturbance timing. Optional STL decomposition and PELT
change-point backends are available for comparison.
