# torch-tracking documentation

```{toctree}
---
hidden: true
---
model_access
```

Welcome to the `torch-tracking` documentation!

`torch-tracking` is deep-learning library for cell tracking in live cell images.

```{note}
This package is a PyTorch reimplementation of the [`deepcell-tracking`][dc-tracking]
library from the TensorFlow-based [`deepcell-tf`][dc-tf] package.

[`deepcell-tf`][dc-tf] is no longer maintained - users are encouraged to use this package instead.
```

```{note}
This package performs **tracking only** - you supply the segmentation masks. It is the
tracking half of the Caliban pipeline; the DynamicNuclearNet segmentation model is not
included here.

See [metrics/PROVENANCE.md][provenance] for how its accuracy compares to the TensorFlow
reference implementation.
```

[dc-tracking]: https://github.com/vanvalenlab/deepcell-tracking
[dc-tf]: https://github.com/vanvalenlab/deepcell-tf
[provenance]: https://github.com/vanvalenlab/torch-tracking/blob/main/metrics/PROVENANCE.md
