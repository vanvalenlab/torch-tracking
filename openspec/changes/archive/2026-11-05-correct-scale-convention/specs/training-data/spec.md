# Spec Delta

## ADDED Requirements

### Requirement: Conversion rescales to the model resolution

Conversion SHALL rescale each movie's images and masks by `mpp / MODEL_MPP`, where `mpp`
is that movie's declared resolution and `MODEL_MPP` is the resolution the model is trained
at. This is the same rule inference applies, so a cell occupies the same number of pixels
in training as at inference. Effective resolution after conversion SHALL be `MODEL_MPP`
for every movie regardless of what it declares.

#### Scenario: A coarser movie is enlarged

- **WHEN** a movie declared at 0.65 µm/px of spatial size `(100, 100)` is converted, with `MODEL_MPP` of 0.55 µm
- **THEN** it SHALL be resized to `(118, 118)`
- **AND** its effective resolution SHALL be 0.55 µm/px

#### Scenario: A movie at the reference resolution is untouched

- **WHEN** a movie declared at `MODEL_MPP` is converted
- **THEN** its scale factor SHALL be 1.0 and its dimensions SHALL be unchanged

#### Scenario: Conversion and inference agree

- **WHEN** the same movie and declared resolution are given to conversion and to inference
- **THEN** both SHALL apply the same scale factor

### Requirement: Both zarrs record each movie's declared resolution

Conversion SHALL write an `mpp` array of shape `(B,)` into both the unprocessed
`<split>.zarr` and the processed `<split>_proc.zarr`, aligned to the movie axis and
holding the resolution each movie declares. Both SHALL be written from a single read of
the conversion-time sidecar, so the two files cannot disagree.

#### Scenario: Conversion persists the resolutions it read

- **WHEN** a split is converted
- **THEN** both zarrs SHALL hold an `mpp` array of shape `(B,)`
- **AND** entry `b` SHALL equal the resolution declared for movie `b`

#### Scenario: Consumers do not reach back to the sidecar

- **WHEN** evaluation or training reads a converted split
- **THEN** it SHALL obtain each movie's resolution from the zarr
- **AND** SHALL NOT require `data-source.npz` to be present

#### Scenario: A stale conversion is rejected

- **WHEN** a zarr produced before this requirement, and therefore holding no `mpp` array, is read
- **THEN** the reader SHALL fail with an error naming the missing array
- **AND** SHALL NOT fall back to a default resolution

### Requirement: The processed zarr holds exactly six arrays

Conversion SHALL write a processed zarr containing `appearances`, `morphologies`,
`centroids`, `labels`, `track_length` and `mpp`, and nothing else. Shapes SHALL be
`(B, T, N, 32, 32, 1)`, `(B, T, N, 3)`, `(B, T, N, 2)`, `(B, T-1, N, N)`, `(B, N, 2)` and
`(B,)` respectively, where `N` is `max_cells`.

#### Scenario: Converting a split writes the documented layout

- **WHEN** a `.trks` split is converted
- **THEN** the processed zarr SHALL hold those six arrays at those shapes
- **AND** SHALL contain no spatial adjacency array

#### Scenario: An unprocessed zarr and a lineage JSON are written alongside

- **WHEN** a `.trks` split is converted
- **THEN** a sibling `<split>.zarr` holding `X`, `y` and `mpp` SHALL be written
- **AND** a sibling `<split>.json` holding the corrected lineages SHALL be written

### Requirement: Processed arrays are chunked along the sampling window

Conversion SHALL chunk each frame-indexed processed array so that one movie's
`track_length`-frame window is a bounded read: one movie on the batch axis,
`track_length` frames on the time axis, every cell slot, and each remaining axis whole.
The time-axis chunk length SHALL be the `track_length` the data was built for, not a
fixed literal.

#### Scenario: A window read touches at most two chunks per array

- **WHEN** one movie's `track_length` consecutive frames are read from a processed array
- **THEN** the read SHALL touch at most two chunks of that array

#### Scenario: Crops are not split across chunks

- **WHEN** `appearances` is chunked
- **THEN** each cell's appearance crop SHALL lie wholly within one chunk
- **AND** no chunk SHALL span more than one movie

#### Scenario: The chunk length follows the configured window

- **WHEN** a split is converted for a `track_length` other than the default
- **THEN** the time-axis chunk length SHALL equal that `track_length`

### Requirement: Training and inference build the spatial graph by the same rule

The dataloader SHALL link two cell slots when both are real cells and their centroid
distance is strictly greater than zero and strictly less than `distance_threshold`.
Padding slots SHALL be excluded from adjacency, matching the rule inference applies, so
no edge connects a real cell to a padding slot.

#### Scenario: Self-links are excluded

- **WHEN** adjacency is computed for a frame
- **THEN** the diagonal SHALL be zero, because a zero distance fails the strict lower bound

#### Scenario: A cell near the origin does not link to padding

- **WHEN** a real cell's centroid lies within `distance_threshold` of `(0, 0)`
- **THEN** that cell SHALL NOT be linked to any padding slot in the frame

#### Scenario: Training and inference agree on the graph

- **WHEN** the same frame is given to the training adjacency rule and the inference adjacency rule
- **THEN** both SHALL produce the same edge set

## MODIFIED Requirements

### Requirement: Samples are fixed-length windows slid across each movie

The dataloader SHALL emit samples of `track_length` consecutive frames, advancing by
`stride`, for every movie in the split. Window generation for a movie SHALL stop when a
window's end frame exceeds that movie's count of non-empty frames, so every non-empty
frame is reachable by some window and no window extends into padding.

#### Scenario: Windows advance by the configured stride

- **WHEN** `track_length` is 8 and `stride` is 1
- **THEN** successive samples SHALL start one frame apart

#### Scenario: Per-sample label windows are one frame shorter

- **WHEN** a sample spans frames `[s, s+track_length)`
- **THEN** its `labels` SHALL span `[s, s+track_length-1)`

#### Scenario: The final non-empty frames are reachable

- **WHEN** a movie holds `n` non-empty frames
- **THEN** some window SHALL cover frame `n-1`
- **AND** some window SHALL cover frame `n-2`

#### Scenario: No window extends into padding

- **WHEN** a movie holds `n` non-empty frames
- **THEN** no emitted window SHALL include any frame at index `n` or beyond

#### Scenario: The stop rule holds at any stride

- **WHEN** `stride` is greater than 1
- **THEN** window generation SHALL still stop at the movie's extent
- **AND** SHALL NOT depend on a window's end frame landing on a particular index

### Requirement: Per-movie resolution comes from an undocumented sidecar

**Partially-fixed:** `correct-scale-convention` removes the runtime half of this coupling;
the conversion-time half remains. Conversion SHALL read each movie's declared resolution
from a `data-source.npz` file located beside the `.trks` file and keyed by split name,
taking the third column of that split's metadata array. Conversion SHALL fail if the
sidecar is absent. This coupling is documented nowhere; it is divergence 20, now narrowed
to conversion alone because the resolutions are persisted into both zarrs.

#### Scenario: The sidecar is missing

- **WHEN** a `.trks` file is converted without `data-source.npz` beside it
- **THEN** conversion SHALL fail rather than fall back to a default resolution

#### Scenario: Training and evaluation no longer need the sidecar

- **WHEN** a converted split is used for training or evaluation
- **THEN** `data-source.npz` SHALL NOT be read

## REMOVED Requirements

### Requirement: Rescaling uses the reciprocal of the inference convention

**Reason**: This is divergence 12, the defect this change exists to correct. Conversion
rescaled by `0.55 / mpp`, the reciprocal of the inference rule, giving an effective
resolution of `mpp² / 0.55` rather than the reference. It is replaced by "Conversion
rescales to the model resolution".

**Migration**: Every `_proc.zarr` produced under the old rule is invalid and must be
regenerated. Weights trained on data converted under the old rule are superseded and
must not be used with corrected conversion — the scale mismatch reappears inverted and
larger than before. The characterization pinning of this behavior is deleted with it.

### Requirement: The processed zarr holds exactly five arrays

**Reason**: The processed zarr now also holds `mpp`, so the array count and the sibling
layout both change. Replaced by "The processed zarr holds exactly six arrays".

**Migration**: Regenerate processed zarrs. A zarr with five arrays and no `mpp` is a
pre-change conversion and is rejected rather than read.

### Requirement: Training adjacency admits edges to padding nodes

**Reason**: This is divergence 17. Training linked real cells to padding slots at the
coordinate origin while inference excluded them — 9,247 real↔padding edges across the test
split, 2.3% of all edges. The register records that divergences 12 and 17 compound, so
correcting the scale without the graph rule would leave half the disagreement in place.
Replaced by "Training and inference build the spatial graph by the same rule".

**Migration**: Requires a retrain; it changes the graph the model is trained on. The
characterization test pinning this behavior is deleted in this change, per the project's
standing rule that a change fixing a pinned divergence deletes the test that pinned it.
