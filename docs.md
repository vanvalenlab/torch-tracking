# Cell Tracking

---

## 1. `CellTracker` - `tracker.py`

### 1.1 Module overview

The `CellTracker` module is the object where all the steps of cell tracking happens. `CellTracker` takes in a time-lapse movie of nuclei to track and their corresponding labeled masks. Features about each object (appearance, morphology, centroid) are extracted: appearance is a 32x32 crop of each nucleus; morphology contains eccentricity, area, and perimeter information; and centroid specifies where in the field of view the cell is located. These features are used to build the embeddings of each cell (a high-dimensional representation of the cells, taking into account their morphology, location, and appearance) using the `CellTracker.get_embeddings()` method. For each frame of the time-lapse, `CellTracker.inference_forward()` will analyze a 7-frame history of the movie and the curent frame. It then returns a prediction of whether a cell and a track are the same, different, or have a mother/daughter cell relationship. The model then assigns tracks to cells using the Hungarian algorithm, which assigns one cell to one track to minimize cost.

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
  - `track_length`: number of historical frames fed to the `InferenceBranch` LSTM — shorter histories are zero-padded
  - `mpp`: microns-per-pixel of the input data; the data is rescaled so that `model_mpp / mpp` matches the resolution the model was trained on

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
- Returns a dict mapping a sequential index (not track ID) to a `(track_length, feature_dim)` array

##### `_update_tracks(assignments, frame, predictions)`

- Iterates over the LAP solution row by row:
  - If `track_idx` maps to an existing track → appends frame, label, embedding, centroid to that track's history and updates `self.y`
  - If `track_idx` is out of range of existing tracks → a birth occurred; calls `_create_new_track()` and then `_get_parent()` to test for division
- After processing all assignments, iterates over tracks with daughters and calls the division-correction logic to cap parent tracks and reroute any incorrect post-division assignments into new daughter tracks

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

- Script that loads a test zarr dataset, runs `CellTracker` on each sample, computes `TrackingMetrics`, and aggregates results into a CSV
- Also contains a visualization utility (`create_timelapse_gif`) for qualitative inspection

---

### 2.2 Functions

#### `build_indices(X)`

- **Purpose**: Determines the last valid (non-empty) frame for each sample in a batched array, to avoid running the tracker on zero-padded frames
- **Parameters**: `X` — batched label array of shape `(B, T, H, W, C)`
- **Logic**: sums pixels across spatial and channel dimensions for each frame; the last frame with a non-zero sum is the end frame
- **Returns**: list of `int`, one per batch entry

#### `pretty_print(df)`

- **Purpose**: Prints a formatted summary of division detection performance (precision, recall, F1) aggregated across the full evaluation set
- **Parameters**: `df` — the results DataFrame with columns `correct_division`, `false_positive_division`, `false_negative_division`
- **Note**: computes macro-level (pooled) metrics rather than averaging per-sample, which is the appropriate choice when sample sizes vary

#### `create_timelapse_gif(im1, im2, output_path, fps, titles, cmap)`

- **Purpose**: Saves a side-by-side animated GIF comparing two label movies (typically predicted vs. ground truth) for qualitative review
- **Parameters**:
  - `im1`, `im2`: arrays of shape `(T, H, W, C)` — note channels are not split, the whole slice is passed to `imshow`
  - `output_path`: destination path for the `.gif` file
  - `fps`: playback speed
  - `titles`: tuple of two strings for subplot titles
  - `cmap`: matplotlib colourmap — `'viridis'` recommended for label images, `'gray'` for raw images
- **Side effects**: writes a file to disk and prints the output path; closes the matplotlib figure to free memory

---

### 2.3 `__main__` evaluation loop

#### Setup

- Config dict — document each key: `batch_size`, `n_layers`, `crop_size`, `crop_mode`, `write_movies`
- Output directories (`movies/`, `metrics/`) created if absent
- Model initialisation and checkpoint loading — note that `model_state_dict` is expected in the checkpoint

#### Data loading

- Opens raw data (`test.zarr`, contains nuclear image `X` and segmentation ground truth segmentation image `y`)
- Loads ground-truth lineage from a paired JSON file indexed by batch position

#### Per-sample loop

- `build_indices` is called once on `y` to get all end frames; note it is called inside the loop but the result is constant — this could be moved outside
- `CellTracker` is constructed and `track_cells()` called for each sample
- `get_lineage_dict()` and `y_tracked` are retrieved from the tracker
- Both lineage dicts are re-keyed as `int` before being passed to `TrackingMetrics` — explain why (JSON keys are strings)
- `TrackingMetrics(...).stats` is the raw dict appended to `metrics_out`

#### Post-loop aggregation

- `metrics_out` is collected into a DataFrame
- Division precision, recall, F1, association accuracy, and target effectiveness are computed as derived columns — document the formulas
- Results written to `eval_results.csv`
- `pretty_print(df)` called for a terminal summary

#### Optional GIF output

- Controlled by `config['write_movies']`; when enabled, calls `_track_review_dict()` (note the private method access) and `create_timelapse_gif` per sample

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
