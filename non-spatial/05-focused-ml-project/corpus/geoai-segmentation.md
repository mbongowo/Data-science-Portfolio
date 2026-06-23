# GeoAI Segmentation — U-Net building footprint segmentation

GeoAI Segmentation trains a semantic segmentation model to extract building
footprints from satellite imagery. The architecture is a U-Net with a ResNet-34
encoder, trained on SpaceNet data with deterministic seeding, Hydra
configuration and MLflow experiment tracking. It reports segmentation quality
with IoU, F1 and Cohen's kappa.

The metrics and image-tiling code are pure numpy and run offline, so they are
testable in CI, while the full GPU training loop needs PyTorch and a GPU. The
project shows a reproducible deep-learning workflow end to end, from tiling
through training to evaluation.
