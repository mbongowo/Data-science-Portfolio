# Segment Geospatial — SAM masking and building quantification

Segment Geospatial applies the Segment Anything Model (SAM) to satellite and
aerial imagery of Douala, Cameroon. SAM's automatic mask generator produces
object masks, and a pure-numpy post-processing core then quantifies them:
connected-component labelling separates objects, area filtering removes noise,
and pixel areas are converted to square metres and hectares to count and size
buildings and fields.

The heavy SAM inference runs on Colab or a GPU notebook, while the
quantification core is light and testable. The project shows how to turn raw
segmentation masks into measured, real-world building and field statistics.
