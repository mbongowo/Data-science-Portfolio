# TorchGeo Landcover — transfer learning for land-cover classification

TorchGeo Landcover classifies whole image patches into land-cover classes in the
EuroSAT style. It runs a controlled comparison: a pure-numpy softmax baseline
that needs no GPU, against a TorchGeo ResNet18 fine-tuned both from ImageNet
pretrained weights and from scratch. The point is to measure how much transfer
learning from pretrained weights helps over training from scratch.

The pipeline covers featurisation, stratified train/test splits and identical
metrics across all three models, so the comparison is apples to apples. The
numpy baseline keeps a runnable, GPU-free path for CI.
