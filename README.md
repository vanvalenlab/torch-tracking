# torch-tracking

A PyTorch implementation of a graph neural network (GNN) cell tracker for live-cell fluorescence time-lapse microscopy. The model assigns consistent track IDs to nuclei across frames, detects cell divisions (mitosis), and handles cell births and deaths.

This is a PyTorch port of the cell tracking model from [Caliban](https://www.biorxiv.org/content/10.1101/803205v3), originally implemented in TensorFlow.

> **Note:** this package performs tracking only — you supply the segmentation masks. See
> [VISION.md](VISION.md) for the project's scope, how it compares to the TensorFlow
> reference, and the constraints that govern changes to it.

---

## Table of Contents

- [torch-tracking](#torch-tracking)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Data Format](#data-format)
    - [Input Arrays](#input-arrays)
    - [Preprocessing: `.trk` → Zarr](#preprocessing-trk--zarr)
    - [Processed Zarr Layout](#processed-zarr-layout)
  - [Basic Usage](#basic-usage)
    - [Running the Tracker](#running-the-tracker)
    - [Outputs](#outputs)
  - [Training](#training)
    - [Training Hyperparameters](#training-hyperparameters)
  - [Inference Hyperparameters](#inference-hyperparameters)
  - [Evaluation](#evaluation)
  - [Model Architecture](#model-architecture)

---

## Installation

```bash
git clone https://github.com/vanvalenlab/torch-tracking.git
cd torch-tracking
pip install .
pip install -r requirements.txt
```

Default model weights are downloaded from [Hugging Face](https://huggingface.co/vanvalenlab). Access is gated: you must join the `vanvalenlab` organization, accept the license terms, and create a Read token exposed as `HF_TOKEN` in your environment.

```bash
export HF_TOKEN="<your-token-here>"
```

With that set, `CellTracker()` downloads the weights on first use and caches them under `.deepcell` in your home directory. **See [doc/model_access.md](doc/model_access.md) for the authoritative instructions**, including license terms — use of the pre-trained weights is restricted to non-commercial academic purposes.

>[!note]
If you plan to use a GPU for inference (which we suggest), you will have to make sure the PyTorch version installed is compatible with the CUDA version on your GPUs. If you are having problems with installation, check out [this page](https://pytorch.org/get-started/previous-versions/) to make sure you installed the correct PyTorch version that is compatible with your system's CUDA version. If you find that there is a version mismatch, follow the instructions on the page to `pip` install the correct PyTorch version.

---

## Data Format

### Input Arrays

The tracker operates on two numpy arrays:

| Array | Shape | dtype | Description |
| --- | --- | --- | --- |
| `movie` | `(T, H, W, C)` | `float32` | Raw fluorescence image. `C` should be `1`. |
| `annotation` | `(T, H, W, 1)` | `int` | Nuclear segmentation mask. Each integer label corresponds to one nucleus; background is `0`. |

- `T` — number of frames
- `H`, `W` — spatial dimensions (pixels)
- `C` — number of channels (always `1` for nuclear images)

The tracker internally rescales images to a resolution of **0.55 µm/pixel**. If your data has a different resolution, set `mpp` when initializing the tracker.

---

### Preprocessing: `.trk` → Zarr

Training data must be converted from `.trk`/`.trks` files (a tar-based format from Caliban) to Zarr arrays before training. Run:

```bash
python -m torch_tracking.preprocess
```

By default this scans `~/.deepcell/tracking/` for `*.trks` files. For each file it produces:

- `<split>.zarr` — raw arrays (`X`, `y`) at native resolution, plus `mpp`
- `<split>_proc.zarr` — pre-extracted features ready for the dataloader
- `<split>.json` — ground-truth lineage records

The `convert_trk_to_zarr` function in [torch_tracking/preprocess.py](torch_tracking/preprocess.py) can also be called directly:

```python
from torch_tracking.preprocess import convert_trk_to_zarr

convert_trk_to_zarr('path/to/train.trks', out_dir='data/')
```

---

### Processed Zarr Layout

The processed Zarr file (`*_proc.zarr`) stores pre-extracted per-cell features:

| Key | Shape | Description |
| --- | --- | --- |
| `appearances` | `(B, T, N, 32, 32, 1)` | 32×32 image crop centered on each cell nucleus |
| `morphologies` | `(B, T, N, 3)` | Per-cell morphological features: area, perimeter, eccentricity |
| `centroids` | `(B, T, N, 2)` | Y and X centroid coordinates in pixels |
| `labels` | `(B, T-1, N, N)` | Temporal adjacency matrix (ground truth): `1` = same cell, `2` = daughter, `0` = different cell, `-1` = padding |
| `track_length` | `(B, N, 2)` | Allocated but never populated (all zeros) |
| `mpp` | `(B,)` | Each movie's declared resolution in µm/px, as float64 |

- `B` — number of movies in the batch
- `T` — number of frames
- `N` — maximum number of cells across the dataset (padded with zeros)

Features are extracted after each movie is rescaled by `mpp / 0.55`, so every cell is
seen at 0.55 µm/px whatever resolution its movie declares. Both zarrs record `mpp`, and
anything reading a split without it fails rather than assuming a resolution; a zarr
converted before this was recorded must be regenerated. The frame-indexed arrays are
chunked as one movie by one `track_length` window (`convert_trk_to_zarr(...,
track_length=8)`), so reading a training window touches at most two chunks.

---

## Basic Usage

### Running the Tracker

```python
import numpy as np
from torch_tracking.tracker import CellTracker

# Load your movie and segmentation masks
movie = np.load('movie.npy')        # shape (T, H, W, 1), float32
annotation = np.load('masks.npy')   # shape (T, H, W, 1), int

# Initialize the tracker
tracker = CellTracker(
    checkpoint_dir='path/to/best_model.pt',  # omit to use ~/.deepcell default
    device='cuda',
    mpp=0.65,        # set to your image's microns-per-pixel
)

# Preprocess and embed the movie
tracker.preprocess_movie(movie=movie, annotation=annotation)

# Run tracking
tracker.track_cells()

# Retrieve results
y_tracked = tracker.y_tracked        # (T, H, W, 1) integer label array
lineage = tracker.get_lineage_dict() # dict of track metadata
df = tracker.dataframe()             # pandas DataFrame summary
```

### Outputs

**`tracker.y_tracked`** — shape `(T, H, W, 1)`. An integer label array where each unique integer corresponds to a tracked cell. Labels are consistent across frames: the same integer means the same cell.

**`tracker.get_lineage_dict()`** — returns a dictionary keyed by track label:

```python
{
    1: {
        'label': 1,
        'frames': [0, 1, 2, 3, ...],   # frames this cell appeared in
        'parent': None,                  # parent track label (if from division)
        'daughters': [4, 5],            # daughter track labels (if divided)
        'frame_div': 12,                # frame where division occurred
        'capped': True                  # True if track ended due to division
    },
    ...
}
```

**`tracker.dataframe()`** — returns a pandas DataFrame with columns `label`, `daughters`, `frame_div`. Optional keyword arguments (`cell_type`, `set`, `part`, `montage`) add metadata columns.

---

## Training

Training requires processed Zarr files (see [Preprocessing](#preprocessing-trk--zarr)).

```python
from pathlib import Path
from torch_tracking.model import GNNTrackingModel
from torch_tracking.loader import create_trk_dataloaders
from torch_tracking.training import Trainer
from torch_tracking.utils import create_optimizer, create_scheduler

config = {
    "optimizer": "radam",
    "learning_rate": 1e-3,
    "weight_decay": 0,
    "scheduler": "reduce_on_plateau",
    "patience": 5,
    "max_epochs": 50,
    "batch_size": 8,
    "n_layers": 2,
    "num_workers": 4,
    "clipnorm": 1.0,
    "loss": "wcce",
    "gamma": 1.0,
    "dropout": 0,
    "device": "cuda:0",
    "data_precision": "bfloat16",
    "stopping_metric": "loss",
    "log_and_save": True,
}

model = GNNTrackingModel(
    graph_layer='gat',
    data_format='channels_last',
    encoder_dim=64,
    n_layers=config['n_layers'],
    crop_size=32,
    dropout=config['dropout'],
)

train_loader, val_loader, _ = create_trk_dataloaders(
    train_path=Path.home() / '.deepcell/tracking/train_proc.zarr',
    val_path=Path.home() / '.deepcell/tracking/val_proc.zarr',
    batch_size=config['batch_size'],
    distance_threshold=89,
    num_workers=config['num_workers'],
)

optimizer = create_optimizer(model, config)
scheduler = create_scheduler(optimizer, config)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=config['device'],
    checkpoint_dir='./checkpoints/',
    log_dir='./logs/',
    max_epochs=config['max_epochs'],
    gradient_clip=config['clipnorm'],
    loss=config['loss'],
    gamma=config['gamma'],
    class_weights=[1, 10, 100],   # upweight mitosis class
    data_precision=config['data_precision'],
    stopping_metric=config['stopping_metric'],
    config=config,
)

trainer.train()
```

Checkpoints are saved to `./checkpoints/<timestamp>/best_model.pt`. TensorBoard logs are written to `./logs/<timestamp>/`.

### Training Hyperparameters

| Parameter | Default | Description |
| --- | --- | --- |
| `learning_rate` | `1e-3` | Initial learning rate for the optimizer |
| `optimizer` | `"radam"` | Optimizer type. Options: `"radam"`, `"adam"`, `"sgd"` |
| `scheduler` | `"reduce_on_plateau"` | LR scheduler. Options: `"reduce_on_plateau"`, `"step"`, `"cosine"` |
| `patience` | `5` | Epochs without improvement before LR is reduced (ReduceLROnPlateau) |
| `max_epochs` | `50` | Maximum number of training epochs |
| `batch_size` | `8` | Number of sequences per batch |
| `clipnorm` | `1.0` | Maximum gradient norm for clipping. Lower values (e.g. `0.001`) stabilize training if gradients are exploding |
| `n_layers` | `2` | Number of GNN message-passing layers in the neighborhood encoder |
| `encoder_dim` | `64` | Feature dimension for all encoders and the GNN hidden state |
| `dropout` | `0` | Dropout rate in the decoder |
| `loss` | `"wcce"` | Loss function. `"wcce"` = weighted cross-entropy; `"focal"` = focal loss |
| `gamma` | `1.0` | Focal loss exponent (only used when `loss="focal"`) |
| `class_weights` | `[1, 10, 100]` | Per-class weights for `[no-link, same-cell, mitosis]`. Mitosis events are rare so a high weight (e.g. `100`) is recommended |
| `data_precision` | `"bfloat16"` | Mixed-precision training mode. Options: `"bfloat16"`, `"float16"`, `"float32"` |
| `distance_threshold` | `64` | Pixel radius for GNN edges (cells farther apart are disconnected). Should match the value used at inference |
| `track_length` | `8` | Number of frames per training window. Should match the inference `track_length` |
| `stride` | `1` | Temporal stride when sliding the training window |

---

## Inference Hyperparameters

These are set on the `CellTracker` object and control the post-processing decisions made after the model scores each potential linkage.

| Parameter | Default | Description |
| --- | --- | --- |
| `death` | `0.999` | Minimum model confidence to call a track "dead" (cell exited the field of view). Higher values make deaths rarer |
| `birth` | `0.999` | Minimum model confidence to call a detection a "new cell" rather than assigning it to an existing track. Higher values make spontaneous births rarer |
| `division` | `0.5` | Minimum model confidence for a new-cell event to be attributed to a cell division (mitosis). Lower values increase division sensitivity at the cost of more false positives |
| `distance_threshold` | `64` | Pixel radius for GNN adjacency at inference. Should match the value used during training |
| `track_length` | `8` | Number of historical frames used to build the LSTM context for each track. Longer histories give more context but require more memory |
| `mpp` | `0.55` | Microns per pixel of your input movie. The model was trained at `0.55 µm/pixel`; images are rescaled to match this before inference |
| `device` | `"cuda"` | PyTorch device string (`"cuda"`, `"cuda:1"`, `"cpu"`, `"mps"`) |

**Tuning guidance:**

- `division` is the most impactful parameter. Run `python -m torch_tracking.eval_sweep` to sweep over values (e.g. `0.3`–`0.8`) and pick the threshold that maximizes division F1 on a held-out set.
- `birth` and `death` rarely need to be changed from `0.999` unless you see many spurious track births or premature track terminations.
- `mpp` must be set correctly for your microscope. An incorrect value shifts the spatial scale of appearance crops and degrades accuracy.

---

## Evaluation

To evaluate on a test set and produce metrics:

```bash
python -m torch_tracking.eval \
    --checkpoint path/to/best_model.pt \
    --split-dir ~/.deepcell/tracking
```

This reads `<split-dir>/test.zarr` and `<split-dir>/test.json` and writes per-movie
results to `metrics/eval_results.csv`. Pass `--output-dir` to write elsewhere, `--split`
to evaluate a split other than `test`, and `--device` to select a device; evaluation runs
on CPU by default and never requires a GPU.

The harness is importable, so the same evaluation is available from Python:

```python
from torch_tracking.eval import evaluate

result = evaluate(checkpoint_path="best_model.pt", split_dir="~/.deepcell/tracking")
print(result.pooled)          # pooled division F1, aa_accuracy, te_accuracy
result.per_movie              # one row per movie
```

Reported metrics include:

- **Division precision / recall / F1** — how accurately the tracker detects mitotic events
- **`aa_accuracy`** — association accuracy (fraction of cells correctly linked frame-to-frame)
- **`te_accuracy`** — target effectiveness, after Hayashida et al. 2020: the fraction of each
  ground-truth track's cells captured by the single result track that best covers it. Despite
  the column name it is not an accuracy of track *ends*

All three are pooled across movies by summing each metric's numerator and denominator and
dividing once — not by averaging per-movie ratios, which gives a different answer.

To sweep over post-processing thresholds and find the best combination:

```bash
python -m torch_tracking.eval_sweep \
    --checkpoint path/to/best_model.pt \
    --split-dir ~/.deepcell/tracking
```

Results are saved to `metrics/postprocess_sweep.csv`.

---

## Model Architecture

The model has two phases: a shared **neighborhood encoder** and separate **training** and **inference** branches.

**Neighborhood Encoder** (`NeighborhoodEncoder`)

Each cell in each frame is described by three features:

- **Appearance** — a 32×32 grayscale crop around the nucleus, encoded by a CNN (`AppearanceEncoder`)
- **Morphology** — area, perimeter, eccentricity, encoded by an MLP (`MorphologyEncoder`)
- **Centroid** — Y/X position in pixels, encoded by an MLP (`CentroidEncoder`)

The three embeddings are summed and passed through a graph attention network (GAT). Edges connect any two cells within `distance_threshold` pixels of each other, allowing the model to learn context from neighboring cells.

**Training Branch** (`TrainingBranch`)

Operates on a window of `track_length` frames. Splits the sequence into "current" (frames `0:T-1`) and "future" (frames `1:T`) embeddings. An LSTM integrates the current embeddings over time. Pairwise centroid differences (within-frame and between-frame) are encoded by `DeltaEncoder` modules and a second LSTM. All features are merged and passed to the decoder, which predicts a temporal adjacency matrix (TAM) of shape `(T-1, N_current, N_future, 3)` where the 3 classes are: different cell, same cell, mitosis.

**Inference Branch** (`InferenceBranch`)

Accepts pre-computed embeddings for existing tracks (from frames up to `T-1`) and embeddings for cells in the new frame `T`. Applies the same LSTM and delta encoding, then decodes to produce linking probabilities. The `CellTracker` solves a Linear Assignment Problem (Hungarian algorithm) on the resulting cost matrix to assign detections to tracks frame by frame.
