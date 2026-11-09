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
    - [What a run writes](#what-a-run-writes)
    - [Choosing the best epoch](#choosing-the-best-epoch)
    - [Training Settings](#training-settings)
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
- `<split>_proc.zarr` — pre-extracted features and the lineage, ready for the dataloader
- `<split>.json` — ground-truth lineage records

Each movie's resolution is read from a `data-source.npz` beside the `.trks` file, or passed
as `mpps=`. See [docs.md](docs.md) §2.4 for the layout.

The `convert_trk_to_zarr` function in [torch_tracking/preprocess.py](torch_tracking/preprocess.py) can also be called directly:

```python
from torch_tracking.preprocess import convert_trk_to_zarr

convert_trk_to_zarr('path/to/train.trks', out_dir='data/', mpps=[0.55, 0.55])  # one per movie
```

---

### Processed Zarr Layout

The processed Zarr file (`*_proc.zarr`) stores pre-extracted per-cell features, and each
movie's corrected lineage in its attributes. The training targets (`labels`: `1` = same
cell, `2` = daughter, `0` = different cell, `-1` = padding) are built from that lineage by
the dataloader, window by window, and are not stored:

| Key | Shape | Description |
| --- | --- | --- |
| `appearances` | `(B, T, N, 32, 32, 1)` | 32×32 image crop centered on each cell nucleus |
| `morphologies` | `(B, T, N, 3)` | Per-cell morphological features: area, perimeter, eccentricity |
| `centroids` | `(B, T, N, 2)` | Y and X centroid coordinates in pixels |
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
    checkpoint_path='path/to/best_model.pt',  # omit to use ~/.deepcell default
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

Training needs processed Zarr files (see [Preprocessing](#preprocessing-trk--zarr)): a directory holding `train_proc.zarr` and `val_proc.zarr`.

Launch a run from the command line. No source file needs editing:

```bash
# On a shared machine, say which card is yours: the default device "cuda" is card 0.
export CUDA_VISIBLE_DEVICES=0

torch-tracking-train --data-dir ~/.deepcell/tracking
# the same thing without the installed command:
python -m torch_tracking.training --data-dir ~/.deepcell/tracking
```

`--data-dir` has no default, because there is no location that is right on every machine. Every setting in the [table below](#training-settings) is also a flag, and `--help` lists them. For example, a shorter run with a different loss:

```bash
torch-tracking-train --data-dir ~/.deepcell/tracking --max-epochs 30 \
    --loss focal_wcce --class-weights 1 10 100 --checkpoint-dir ./my_run/
```

From Python, the same run is a configuration object and one call:

```python
from torch_tracking.training import TrainingConfig, run_training

config = TrainingConfig(data_dir="~/.deepcell/tracking", max_epochs=30, device="cuda")
trainer = run_training(config)   # builds the model and dataloaders, trains, returns the Trainer
```

To assemble the pieces yourself, `Trainer(model, train_loader, val_loader, config)` takes exactly those four arguments. The configuration is a `TrainingConfig`, not a dictionary, and the trainer builds its own optimizer and scheduler from it.

**Memory.** Each dataloader worker holds its own copy of the training arrays in RAM, about 15 GB per worker for DynamicNuclearNet-tracking, so the default `--num-workers 4` needs on the order of 60 to 80 GB of RAM. Lower it on a busy machine.

### What a run writes

Beneath `--checkpoint-dir` (default `./checkpoints/`), in a folder named for the start time:

| File | Contents |
| --- | --- |
| `config.json` | Every setting the run used, including the loss class weights |
| `checkpoint_epoch_NNNN.pt` | One per finished epoch. All are kept unless `--max-checkpoints N` caps them to the most recent N |
| `best_model.pt` | A copy of the checkpoint for the best epoch. Never removed by the cap |
| `training_history.json` | Per-epoch training and validation metrics, plus `best_epoch`, `best_value` and `stopping_metric`. Written when the run ends; if a run is killed, the same lists are in the latest checkpoint |

TensorBoard logs go to `--log-dir` (default `./logs/`). With `--no-log-and-save` a run writes none of this, and still runs to the end.

To continue an interrupted run, call `trainer.load_checkpoint(path)` and then `trainer.train()`. Training resumes with the epoch *after* the one the checkpoint recorded, and restores the optimizer, the learning-rate schedule, the gradient scaler and the early-stopping counter.

### Choosing the best epoch

`--stopping-metric` names the validation metric that picks the best epoch, drives early stopping and drives the plateau scheduler. The default is `loss`, which is minimized. Every other metric (`accuracy` and the `precision_class_*`, `recall_class_*` and `f1_class_*` keys) is maximized, and the direction is applied consistently to all three. An unrecognized name is rejected when the trainer is built, with the list of valid names.

Validation loss is only a weak ranking of epochs by validation daughter-class F1 (rank correlation about -0.40 across 297 epochs in seven past runs), but that is **not evidence of accuracy left on the table**. Traced over the 444 ground-truth divisions of the `val` split, a *perfect* daughter-class head would raise pooled division F1 by about 0.006, against a sampling standard error of about 0.012. Which epoch is kept barely moves the metric this project is judged on, so the default stays validation loss. Every epoch is kept by default, so that claim can be retested against the real evaluation.

**Reproducing a run.** Pass `--seed N` and the same configuration gives the same recorded metrics on CPU (GPU kernels can still differ in the last bits). With no `--seed`, a seed is drawn and written to `config.json`, so the run can be repeated afterwards:

```bash
torch-tracking-train --data-dir ~/.deepcell/tracking --seed 1234
```

**What a recorded loss means.** The loss is the weighted cross-entropy per graded entry. Every loss recorded before `correct-checkpoint-selection` is **one third of the per-entry cross-entropy**, because it was also averaged over the three classes: multiply an earlier loss by 3 to compare it with a later one. Earlier validation losses were also computed at full precision while training ran in reduced precision; validation now runs at the same precision as training, so the two losses of an epoch come from one numeric path. Neither change moves the epoch a run selects.

**Gradient norm.** Each epoch's training history records `grad_norm` (mean gradient norm before clipping), `grad_norm_max`, and `clip_engaged` (whether any batch reached `--clipnorm`), so whether clipping ever mattered is answered by the run itself.

### Training Settings

Every field of `TrainingConfig`. Each is recorded in `config.json`.

| Setting (`--flag`) | Default | Description |
| --- | --- | --- |
| `data_dir` (`--data-dir`) | *required* | directory holding train_proc.zarr and val_proc.zarr (required) |
| `optimizer` (`--optimizer`) | `"radam"` | one of radam, adamw |
| `scheduler` (`--scheduler`) | `"reduce_on_plateau"` | one of reduce_on_plateau, cosine, step, exp, none |
| `device` (`--device`) | `"cuda"` | torch device; on a shared machine pick the card with CUDA_VISIBLE_DEVICES |
| `checkpoint_dir` (`--checkpoint-dir`) | `"./checkpoints/"` | parent directory for per-run checkpoint folders |
| `log_dir` (`--log-dir`) | `"./logs/"` | parent directory for per-run TensorBoard logs |
| `step_size` (`--step-size`) | `5` | epochs between decays under the 'step' scheduler |
| `step_gamma` (`--step-gamma`) | `0.1` | learning-rate multiplier at each decay under the 'step' scheduler |
| `max_epochs` (`--max-epochs`) | `50` | number of epochs to train |
| `num_workers` (`--num-workers`) | `4` | dataloader worker processes; each holds its own copy of the training set in RAM |
| `n_layers` (`--n-layers`) | `2` | graph layers in the model |
| `clipnorm` (`--clipnorm`) | `1.0` | gradient-norm clip; 0 disables clipping |
| `batch_size` (`--batch-size`) | `8` | windows per batch |
| `learning_rate` (`--learning-rate`) | `0.001` | initial learning rate |
| `patience` (`--patience`) | `5` | epochs without improvement before 'reduce_on_plateau' decays |
| `early_stopping_patience` (`--early-stopping-patience`) | `10` | epochs without improvement before stopping (with --enable-early-stopping) |
| `enable_early_stopping` (`--enable-early-stopping`) | `False` | stop when the stopping metric stops improving |
| `log_and_save` (`--log-and-save`) | `True` | write checkpoints, history and TensorBoard logs; off writes nothing |
| `class_weights` (`--class-weights`) | `[1, 10, 100]` | loss weight per class, in [different, same, mitosis] order |
| `loss` (`--loss`) | `"wcce"` | one of wcce, focal, focal_wcce |
| `stopping_metric` (`--stopping-metric`) | `"loss"` | validation metric that selects the best epoch and drives early stopping; loss is minimized, every other metric maximized |
| `gamma` (`--gamma`) | `1.0` | focusing parameter of the focal loss modes |
| `truncate_dataset` (`--truncate-dataset`) | `None` | use only the first N movies (for smoke tests) |
| `data_precision` (`--data-precision`) | `"bfloat16"` | one of bfloat16, float16, float32 |
| `dropout` (`--dropout`) | `0` | dropout rate in the model |
| `crop_size` (`--crop-size`) | `32` | appearance crop size the model expects |
| `decay` (`--decay`) | `0.99` | learning-rate multiplier per epoch under the 'exp' scheduler |
| `weight_decay` (`--weight-decay`) | `0.0` | weight decay passed to the optimizer |
| `max_checkpoints` (`--max-checkpoints`) | `None` | keep only the most recent N per-epoch checkpoints; default keeps every epoch |
| `seed` (`--seed`) | `None` | seed for every random stream the run draws on; a seed is drawn and recorded when none is given, so the run can still be repeated |

Three further values are fixed by the entry point rather than configured: `distance_threshold` (89, `utils.DISTANCE_THRESHOLD`), `track_length` (8) and `stride` (1) are dataloader arguments. `distance_threshold` and `track_length` must match the values used at inference.

`class_weights` applies in `wcce` and `focal_wcce`. The `focal` loss has no use for weights: asking for them from `TrackingLoss` directly is an error, and the trainer prints a note that the recorded weights have no effect.

---

## Inference Hyperparameters

These are set on the `CellTracker` object and control the post-processing decisions made after the model scores each potential linkage.

| Parameter | Default | Description |
| --- | --- | --- |
| `death` | `0.999` | Minimum model confidence to call a track "dead" (cell exited the field of view). Higher values make deaths rarer |
| `birth` | `0.999` | Minimum model confidence to call a detection a "new cell" rather than assigning it to an existing track. Higher values make spontaneous births rarer |
| `division` | `0.5` | Minimum model confidence for a new-cell event to be attributed to a cell division (mitosis). Lower values increase division sensitivity at the cost of more false positives |
| `distance_threshold` | `89` | Pixel radius for GNN adjacency at inference. Should match the value used during training |
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

- **Division precision / recall / F1** — how often a true division is reproduced exactly: the
  same parent *and* the same daughter set. A division whose parent is found but whose daughters
  are wrong or incomplete (a "mismatch") counts against both recall and precision, as a missed
  division and as a false positive. Earlier versions of this package dropped mismatches from the
  score entirely, which flattered division F1 (on the recorded test split, 0.9588 instead of
  0.9157) without any change to the tracker
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
