# Cell Tracking

---

## 1. `CellTracker` - `tracker.py`

### 1.1 Module overview

The `CellTracker` module is the object where all the steps of cell tracking happens. `CellTracker` takes in a time-lapse movie of nuclei to track and their corresponding labeled masks. Features about each object (appearance, morphology, centroid) are extracted: appearance is a 32x32 crop of each nucleus; morphology contains eccentricity, area, and perimeter information; and centroid specifies where in the field of view the cell is located. These features are used to build the embeddings of each cell (a high-dimensional representation of the cells, taking into account their morphology, location, and appearance) using the `CellTracker.get_embeddings()` method. For each frame of the time-lapse, `CellTracker.inference_forward()` will analyze the previous `track_length` frames of each track (8 by default) together with the current frame. It then returns a prediction of whether a cell and a track are different, the same, or have a mother/daughter cell relationship, in that order. The model then assigns tracks to cells using the Hungarian algorithm, which assigns one cell to one track to minimize cost.

After tracking the full time-lapse movie, the `CellTracker` object contains the tracked masks (`CellTracker.y_tracked()`), the original movie (`CellTracker.X`), and the original untracked masks (`CellTracker.y`). In addition, the cell lineage information can be returned in the form of a dictionary with the `CellTracker.get_lineage_dict()` method.

---

### 1.2 `CellTracker` class

- **Purpose**: Validates inputs, rescales data to the model's native resolution, extracts per-frame features (appearance, morphology, centroid, adjacency), and pre-computes GNN embeddings for the full movie before tracking begins

- **Parameters**:
- `tracking_model`: a `GNNTrackingModel` instance — will be moved to `device` and set to eval mode
  - `device`: `'cuda'` or `'cpu'`
  - `distance_threshold`: pixel radius used to define edges in the adjacency graph — cells farther apart than this are not connected
  - `death`: diagonal cost used in the death sub-matrix of the LAP — higher values make track termination less likely
  - `birth`: diagonal cost used in the birth sub-matrix of the LAP — higher values make new track creation less likely
  - `division`: minimum class-2 (daughter) probability required for a new track to be labelled a division rather than a fresh birth
  - `track_length`: number of preceding frames of each track fed to the `InferenceBranch` LSTM — a younger track is padded by repeating its most recent frame. The default of 8 is the window the processed training data is built for, so change the two together
  - `mpp`: microns-per-pixel of the input data; the data is rescaled by `mpp / model_mpp` (`model_mpp` is 0.55), so each pixel spans the resolution the model was trained on. A movie coarser than the model is enlarged

- **Raises**: `ValueError` for rank != 4, mismatched spatial shapes

- **State initialised**: `self.tracks` dict, feature arrays, embedding cache, `id_to_idx` / `idx_to_id` mappings, rescaled `self.X` and `self.y`

---

#### Public methods

##### `preprocess_movie()`

- **Purpose**: Extract features from each object in the image and preprocess embeddings for tracking

- **Parameters**:
  - `movie`: raw time-series NumPy array `(T, Y, X, C)`
  - `annotation`: segmentation label array, must match `movie` shape except in the channel dimension
  
##### `track_cells()`

- **Purpose**: runs the full tracking loop
- **Workflow**:
  1. Calls `_initialize_tracks()` to seed one track per cell in frame 0
  2. Iterates over frames 1 … T−1, calling `_track_frame()` at each step
- **Side effects**: populates `self.tracks`, builds `self.y_tracked` frame by frame, appends per-frame cost matrices to `self.a_matrix` and `self.c_matrix`
- **Returns**: nothing — results are accessed via `get_lineage_dict()` or `dataframe()` afterwards

##### `get_lineage_dict()`

- **Purpose**: Serialises the completed tracks into the standard lineage dictionary format expected by `.trk` files and `TrackingMetrics`
- **Returns**: `dict` keyed by 1-based cell label; each value contains `label`, `frames`, `parent`, `daughters`, `frame_div`, `capped`
- **Note**: since the background of the mask is always 0, `parent` and `daughters` are indexed from 1. To access a track, simply index it with its track ID, which corresponds to its label in the `CellTracker.y_tracked` attribute.

##### `dataframe(**kwargs)`

- **Purpose**: Exports a summary of all tracks as a `pandas.DataFrame`, optionally annotated with dataset-level metadata columns
- **Parameters**: keyword arguments `cell_type`, `set`, `part`, `montage` — any subset may be supplied; each is broadcast as a constant column across all rows
- **Returns**: DataFrame with columns for `label`, `daughters` (as label lists), `frame_div`, plus any supplied metadata columns
- **Raises**: `ValueError` for unrecognised keyword arguments

---

#### Private methods (document the logic, not just the signature)

##### `_clean_labels(annotation)`

- Copies the annotation array and calls `clean_up_annotations` to give every cell a globally unique label across all frames (preventing label collisions that would confuse the tracker)

##### `_extract_features()`

- Iterates over every frame, calling `get_image_features` to obtain appearance crops, morphology vectors `(area, perimeter, eccentricity)`, and centroids for each cell
- Builds the adjacency matrix per frame using Euclidean distance between centroids — two cells are connected if their distance is > 0 and < `distance_threshold`; padded (zero-morphology) nodes are explicitly disconnected
- Returns four zero-padded arrays shaped `(1, T, max_cells, …)`. The padding is there to run the GNN on a GPU. Changing the adjacency matrix size every iteration would increase computation overhead.
- Also populates `id_to_idx` and `idx_to_id` mappings used throughout tracking

##### `_compute_embeddings(appearances, morphologies, centroids, adj_matrices)`

- Runs under `torch.no_grad()`
- Converts numpy arrays to tensors via `_to_tensors()`, then calls `model.get_embeddings()` to produce `(T, N, embedding_dim)` embeddings for the full movie in a single forward pass
- Embeddings are cached in `self.features` and reused at every subsequent frame during tracking — the model is not called again per frame for the history, only for the current frame's cells via `inference_forward`

##### `_initialize_tracks()`

- Creates one track entry in `self.tracks` for each cell in frame 0 via `_create_new_track()`
- Seeds `self.y_tracked` with the first frame's labels

##### `_create_new_track(frame, old_label)`

- Allocates a new entry in `self.tracks` with fields: `label`, `frames`, `frame_labels`, `daughters`, `capped`, `frame_div`, `parent`, `embedding`, `centroid`
- Rewrites the label in `self.y` from `old_label` to the new sequential track label
- **Important**: raises an exception if the new label already exists in any frame > 0, which would indicate a label collision. This should never happen, but may if `_clean_labels()` is malfunctioning.

##### `_track_frame(frame)`

- Orchestrates a single frame step: calls `_get_cost_matrix()`, solves it with `scipy.optimize.linear_sum_assignment` (Hungarian algorithm), then calls `_update_tracks()` with the result

##### `_get_cost_matrix(frame)`

- Runs under `torch.no_grad()`
- Assembles the current track histories (`_fetch_tracked_features`) and current-frame cell features (`_get_frame_features`) into tensors
- Calls `model.inference_forward()` to get `(N_tracks, N_cells, 3)` class probabilities
- Derives assignment costs as `1 − P(same cell)` (class 1 probability), and sets cost to 1.0 for any capped (already-divided) track
- Calls `_build_cost_matrix()` to wrap this into the full LAP matrix

##### `_build_cost_matrix(assignment_matrix)`

- Constructs the four-quadrant cost matrix:
  - **Top-left**: the raw `(N_tracks × N_cells)` assignment costs from the model
  - **Top-right** (death): diagonal `self.death`, off-diagonal 1.0 — allows a track to "die" rather than be forced to link
  - **Bottom-left** (birth): diagonal `self.birth`, off-diagonal 1.0 — allows a new cell to be born
  - **Bottom-right** (Mordor): transpose of the assignment matrix — a standard LAP trick to make the problem feasible. Assignments will never be made in this region.
- Lower cost = more likely to be assigned

##### `_fetch_tracked_features(before_frame, feature_name)`

- For each active track, collects the most recent `track_length` frames of the specified feature (embedding or centroid) from `self.tracks`
- Pads with the last known value if fewer than `track_length` frames are available (e.g. for newly-created tracks)
- Returns a dict mapping each track's number to a `(track_length, feature_dim)` array. A track with no frame before `before_frame` is absent; the remaining keys still name their own tracks, so a key is never a position in the iteration

##### `_update_tracks(assignments, frame, predictions)`

- Iterates over the LAP solution row by row:
  - Each assignment pairs a cost-matrix row with a cell. Rows below the number of scored tracks are existing tracks, resolved to a track number through `predictions['track_ids']`: the cell's frame, label, embedding and centroid are appended to that track's history
  - Later rows are birth rows → a birth occurred; calls `_create_new_track()`, which assigns the next track number, and then `_get_parent()` to test for division
- After processing all assignments, iterates over tracks with daughters and calls the division-correction logic to cap parent tracks and reroute any incorrect post-division assignments into new daughter tracks. A parent may have two or more daughters (multipolar mitosis occurs in the data). If rerouting would remove a track's only frame, it raises rather than leave an empty track

##### `_get_parent(frame, cell_id, predictions)`

- Scans all active (non-capped) tracks for any whose class-2 (daughter) probability for `cell_id` exceeds `self.division`
- Returns the track ID with the highest qualifying probability, or `None` if no candidate exceeds the threshold
- Guards against a newly-appeared sibling being labelled as its own parent

##### `_get_frame(tensor, frame)` / `_get_cells_in_frame(frame)`

- Small helpers that respect `data_format` when indexing frames, and return the non-background cell labels in a given frame

##### `_get_feature(frame, cell_id, feature_name)` / `_get_frame_features(frame, feature_name)`

- Look up single-cell or all-cell feature vectors from `self.features` using the `id_to_idx` mapping

##### `_track_review_dict()`

- Packages `self.tracks`, `self.X`, `self.y`, and `self.y_tracked` into a single dict — primarily used for inspection and GIF generation in the evaluation script

---

### 1.3 Track data structure

Document the internal schema of a single entry in `self.tracks`:

| Key | Type | Description |
| --- | --- | --- |
| `label` | `int` | 1-based cell label used in `self.y` |
| `frames` | `list[int]` | Frame indices where this track is active |
| `frame_labels` | `list[int]` | Original cell label in each frame before relabelling |
| `daughters` | `list[int]` | Internal 0-based track IDs of daughter tracks |
| `capped` | `bool` | True after the track has divided and should no longer accept assignments |
| `frame_div` | `int \| None` | Frame index at which division occurred |
| `parent` | `int \| None` | Internal 0-based track ID of the parent track |
| `embedding` | `np.ndarray (K, D)` | Per-frame GNN embedding history |
| `centroid` | `np.ndarray (K, 2)` | Per-frame centroid history |

---

### 1.4 Indexing conventions (potential gotchas)

- Internal track IDs are 0-based; exported labels (in lineage dicts and DataFrames) are 1-based. The conversion happens inside `get_lineage_dict()` and `_track_review_dict()`.
- `id_to_idx` maps `(frame, cell_id)` → position in the padded feature array; `idx_to_id` is the reverse. Both are built during `_extract_features()` and are critical for linking the LAP solution back to cell identities.
- `relevant_tracks` in `_get_cost_matrix` uses sequential indices (0, 1, 2, …) as rows of the assignment matrix; these are mapped back to internal track IDs via the ordered dict produced by `_fetch_tracked_features`.

---

## 2. Evaluation Loop - `eval.py` 

### 2.1 Module overview

- `torch_tracking.eval` exposes `evaluate()`: loads a split, runs `CellTracker` on each
  movie, computes `TrackingMetrics`, pools the results and writes a per-movie CSV.
  Importing the module performs no evaluation.
- `torch_tracking.eval_sweep` exposes `sweep()`, the same evaluation across a grid of
  postprocessing thresholds. It imports the frame-extent rule rather than redefining it.
- Both also provide a `__main__` CLI. Neither reads any path fixed in source.
- `create_timelapse_gif` remains available for qualitative inspection.

---

### 2.2 Functions

#### `evaluate(checkpoint_path, split_dir, ...)`

- **Purpose**: evaluate a checkpoint against a ground-truth split
- **Key parameters**: `checkpoint_path` and `split_dir` are required; `split` (default
  `'test'`), `output_dir` (default `'metrics'`, or `None` to compute without writing),
  `device` (default `'cpu'`), the tracking parameters, and `threshold`
- **Returns**: `EvaluationResult` with `.per_movie` (one row per movie), `.pooled` (the
  three contract metrics) and `.results_path`
- **Raises**: `FileNotFoundError` naming the missing path when the split is absent
- **Side effects**: writes `eval_results.csv` beneath `output_dir`, creating it if needed

#### `pooled_metrics(df)`

- **Purpose**: pool division F1, association accuracy and target effectiveness across movies
- **Parameters**: `df` — any frame carrying the columns in `eval.COUNT_COLUMNS`
- **Logic**: sums each metric's numerator and denominator across movies and divides once.
  A movie whose own ratio is undefined still contributes its counts
- **Note**: this is deliberately *not* the mean of per-movie ratios. On the recorded test
  split, pooling gives association accuracy 0.9728 where averaging gives 0.9715

#### `build_indices(y)`

- **Purpose**: per-movie frame extent, used to avoid running the tracker on zero-padded frames
- **Parameters**: `y` — batched label array of shape `(B, T, H, W, C)`
- **Logic**: counts frames whose mask holds any labelled pixel, via
  `utils.count_nonempty_frames`, and returns that count
- **Returns**: list of `int`, one per movie. It is the exclusive end of the occupied
  frames, so used as a slice bound it includes every non-empty frame. The training
  dataloader derives its extent from the same `count_nonempty_frames`

#### `pretty_print(pooled)`

- **Purpose**: print every contract metric to the terminal
- **Parameters**: `pooled` — the dict returned by `pooled_metrics`, i.e. `result.pooled`.
  It previously took the results DataFrame and reported division metrics only

#### `create_timelapse_gif(im1, im2, output_path, fps, titles, cmap)`

- **Purpose**: saves a side-by-side animated GIF comparing two label movies (typically
  predicted vs. ground truth) for qualitative review
- **Parameters**:
  - `im1`, `im2`: arrays of shape `(T, H, W, C)` — note channels are not split, the whole
    slice is passed to `imshow`
  - `output_path`: destination path for the `.gif` file
  - `fps`: playback speed
  - `titles`: tuple of two strings for subplot titles
  - `cmap`: matplotlib colourmap — `'viridis'` recommended for label images, `'gray'` for
    raw images
- **Side effects**: writes a file to disk and prints the output path; closes the matplotlib
  figure to free memory. Imports matplotlib lazily, so importing `eval` does not

---

### 2.3 The evaluation loop

#### Data loading

- `split_dir` must hold `<split>.zarr` (nuclear image `X` and ground-truth masks `y`) and
  `<split>.json` (ground-truth lineages, indexed by movie position)
- A missing file raises `FileNotFoundError` naming it, rather than failing later

#### Per-movie loop

- `build_indices` is called **once**, outside the loop; the result is constant
- A fresh `CellTracker` is constructed per movie, so no state carries across movies and
  evaluation order cannot affect any result
- Each tracker is constructed with the movie's declared `mpp`, read from the split's
  `mpp` array, so it is rescaled by `mpp / model_mpp` exactly as conversion did. A zarr
  holding no `mpp` raises `ValueError` rather than defaulting to 0.55 µm/px
- Both lineage dicts are re-keyed as `int` before reaching `TrackingMetrics`, because JSON
  object keys are always strings while `get_lineage_dict()` returns `int` keys
- `TrackingMetrics(...).stats` is the raw dict collected per movie

#### Post-loop aggregation

- Rows are collected into a DataFrame; per-movie ratio columns are derived from the counts
- `pooled_metrics` computes the three contract metrics from the count columns
- Results are written to `<output_dir>/eval_results.csv` — the directory created is the
  directory written into
- `pretty_print(result.pooled)` gives a terminal summary of all three

#### Optional GIF output

- Controlled by the `write_movies` argument; when enabled, calls `_track_review_dict()`
  (note the private method access) and `create_timelapse_gif` per movie, beneath `movies_dir`

---

### 2.4 Training data — `preprocess.py` and `loader.py`

#### Conversion: `convert_trk_to_zarr(filename, out_dir=None, track_length=8, mpps=None)`

Turns a `<split>.trks` archive into three files, all written into `out_dir` (which must
already exist) or beside the `.trks` file when `out_dir` is `None`:

- `<split>.zarr` — `X`, `y` and `mpp` at native resolution
- `<split>_proc.zarr` — the arrays the dataloader reads (below)
- `<split>.json` — the corrected lineage of every movie

Both zarrs are written together or not at all: each is staged as `<name>.partial` inside the
output directory and renamed into place once both are complete.

#### Per-movie resolution

Each movie's declared resolution (µm/px) comes from one of two places:

- **`mpps=`** — a sequence with exactly one value per movie. When given, nothing else is read.
- **`data-source.npz`** — otherwise, a file of exactly that name **beside the `.trks`
  file** (not in `out_dir`). It is keyed by split name (`train`, `val`, `test`); each value
  is a per-movie metadata array whose **third column (index 2)** is that movie's resolution.

If neither is available, or the number of resolutions differs from the number of movies,
conversion fails naming what is missing. No default resolution is ever assumed. The values
are stored in both zarrs, so nothing after conversion needs `data-source.npz`.

#### The processed zarr

Exactly four arrays, plus the lineage:

| Key | Shape | Content |
| --- | --- | --- |
| `appearances` | `(B, T, N, 32, 32, 1)` | per-cell crops |
| `morphologies` | `(B, T, N, 3)` | area, perimeter, eccentricity |
| `centroids` | `(B, T, N, 2)` | centroid coordinates |
| `mpp` | `(B,)` | declared resolution, µm/px |

`N` is the largest cell count of any movie; unused slots are all zero. The corrected
lineage of every movie is stored in the zarr's attributes under `lineages`. A processed zarr
without it is rejected by the dataloader with an error naming it; no sibling `.json` is read
in its place. Neither the training targets nor the spatial graph is stored.

#### Training targets are built per window

The dataloader builds each sample's `labels` from the lineage when the sample is drawn,
for just the frames in that window: `(track_length - 1, N, N)`, one matrix per step from a
frame to the next.

Rows are the cells of one frame (the tracks) and columns are the cells of the next frame
(the candidates). An entry is graded only when both of its slots carry a cell: the row's
cell in the first frame and the column's cell in the next.

| Value | Meaning |
| --- | --- |
| `1` | same cell: the cell is in both frames |
| `2` | daughter: from a dividing parent to each daughter, at the parent's final frame |
| `0` | different cell: both slots carry a cell and they are not the same cell |
| `-1` | padding, ignored by the loss: the row has no cell in the first frame, or the column has none in the next |

A slot with no cell, whether unused capacity, a cell not yet born, or a cell that has ended,
holds all-zero features, so all of them are padding. Two consequences are deliberate: a
newborn cell's column is graded in the step into its first frame (it is not any existing
track), and a cell that ends has its row graded in its last frame (it matches no candidate).
The `2` entries are written after padding so it can never erase one.

The test dataloader is unshuffled and unaugmented, like validation. Only the training
loader shuffles and augments.

---

## 3. `Trainer` - `training.py`

### 3.1 Module overview

- Encapsulates the full supervised training loop: forward pass, loss computation, gradient update, validation, learning rate scheduling, early stopping, TensorBoard logging, and checkpoint management
- Wraps a `GNNTrackingModel` and is configured entirely through constructor arguments — no global config file is required at runtime

---

### 3.2 `Trainer` class

#### Constructor — `__init__`

- **Parameters** — document each:
  - `model`: `GNNTrackingModel` instance — moved to `device` in the constructor
  - `train_loader`, `val_loader`: PyTorch `DataLoader` objects producing batches with keys `appearances`, `morphologies`, `centroids`, `adj_matrices`, `labels`
  - `optimizer`: pre-built optimizer (use `create_optimizer` from `utils.py`)
  - `scheduler`: optional LR scheduler (use `create_scheduler` from `utils.py`); `ReduceLROnPlateau` is handled separately from step-based schedulers
  - `device`: device string passed to `torch.amp.autocast` — must match the device the model is on
  - `checkpoint_dir`, `log_dir`: base directories; a timestamp suffix is appended automatically to avoid overwriting runs
  - `max_epochs`: hard ceiling on training duration
  - `gradient_clip`: maximum gradient norm; set to 0 to disable clipping
  - `early_stopping_patience`: number of validation epochs without improvement before stopping (only active if `enable_early_stopping=True`)
  - `enable_early_stopping`: flag to turn early stopping on or off without changing patience
  - `log_and_save`: when `False`, TensorBoard writer is not created and no checkpoints are written — useful for quick debugging runs
  - `config`: arbitrary dict saved as `config.json` alongside checkpoints for reproducibility
  - `loss`: one of `'wcce'`, `'focal'`, `'focal_wcce'` — passed to `TrackingLoss`
  - `class_weights`: optional list of per-class weights passed to `TrackingLoss`; when `None`, weights are computed dynamically from each batch, but leads to unstable training. The default is 1:10:100 for different:same:mitosis.
  - `stopping_metric`: key from the metrics dict used to determine best model — typically `'loss'`
  - `gamma`: focal loss focusing parameter (only used when `loss` contains `'focal'`)
  - `data_precision`: `'float32'`, `'float16'`, or `'bfloat16'` — controls `torch.amp.autocast` dtype and whether `GradScaler` is active
- **Side effects**: creates checkpoint and log directories, writes `config.json`, instantiates `SummaryWriter`, `TrackingLoss`, `MetricsTracker`, and `EarlyStopping`

---

#### `train_epoch()`

- **Purpose**: Runs one full pass over the training set
- **Workflow**:
  1. Sets model to `train()` mode
  2. For each batch: moves tensors to device, calls `model.training_forward()` under `autocast`, computes loss, scales and calls `backward()`, checks for NaN/Inf, unscales gradients, clips norm, steps scaler and optimizer
  3. Updates `MetricsTracker` and refreshes the tqdm progress bar with per-class F1 and loss
- **Returns**: metrics dict from `MetricsTracker.get_metrics()` — resets the tracker before returning
- **Note on precision**: when `data_precision='float32'`, `GradScaler` is instantiated with `enabled=False`, making it a no-op — so the scaler calls are safe regardless of precision mode

#### `validate()`

- **Purpose**: Runs one full pass over the validation set without gradient computation
- Decorated with `@torch.no_grad()` — no need to call `torch.no_grad()` manually inside
- Identical structure to `train_epoch()` except no backward pass, no scaler, and NaN batches are silently skipped
- **Returns**: metrics dict; resets tracker before returning

#### `save_checkpoint(is_best)`

- **Purpose**: Persists model and optimizer state to disk
- Saves the full checkpoint dict (epoch, model state, optimizer state, best val loss, history) to `checkpoint_epoch_N.pt` for regular saves, or `best_model.pt` when `is_best=True`
- Only the best checkpoint overwrites; regular checkpoints are pruned to keep only the last 3

#### `load_checkpoint(checkpoint_path)`

- **Purpose**: Restores a training run from a saved checkpoint
- Restores model weights, optimizer state, epoch counter, best val loss, and full history
- Allows resuming training by calling `train()` after loading

#### `train()`

- **Purpose**: Main training loop — calls `train_epoch()` and `validate()` for each epoch, steps the scheduler, saves checkpoints, and triggers early stopping
- **Scheduler step logic**: `ReduceLROnPlateau` is stepped with the current stopping metric; all other schedulers are stepped without arguments — document that list schedulers (the `'caliban'` case) are not handled
- **Best model logic**: `is_best` is determined by comparing `val_metrics[stopping_metric]` against `self.best_val_loss`; note that `save_checkpoint(is_best=True)` is always called at the end of training, saving whatever state the model is in at the final epoch (not necessarily the best)
- **TensorBoard**: all keys in `train_metrics` and `val_metrics` are logged under `train/` and `val/` namespaces
- **History**: `train_history` and `val_history` are written to `training_history.json` at the end

---

### 3.3 `__main__` example script

- Shows a minimal end-to-end training run using `create_trk_dataloaders`, `create_optimizer`, and `create_scheduler` from `utils.py`
- Document the config keys used and their meaning: `optimizer`, `learning_rate`, `weight_decay`, `decay`, `scheduler`, `max_epochs`, `batch_size`, `n_layers`, `num_workers`, `clipnorm`, `step_size`, `crop_mode`, `patience`, `log_and_save`, `enable_early_stopping`, `crop_size`, `truncate_dataset`, `loss`, `dropout`, `device`, `label_smoothing`, `stopping_metric`, `data_precision`, `gamma`

---

### 3.4 Checkpoints

Document the schema of a saved checkpoint file:

| Key | Description |
| --- | --- |
| `epoch` | Last completed epoch index |
| `model_state_dict` | `model.state_dict()` — load with `model.load_state_dict()` |
| `optimizer_state_dict` | Optimizer state — load with `optimizer.load_state_dict()` |
| `best_val_loss` | Best observed validation metric across all epochs |
| `train_history` | List of per-epoch train metric dicts |
| `val_history` | List of per-epoch val metric dicts |
