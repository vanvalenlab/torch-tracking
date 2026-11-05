# training-data Specification

## Purpose

Converting `DynamicNuclearNet-tracking` `.trks` archives into the processed zarr the
trainer reads, and serving windowed, augmented batches from it. Everything between a
`.trks` file on disk and the tensors handed to the training loop.

## Requirements

### Requirement: Archive loading accepts both lineage key spellings

The system SHALL read a `.trks` archive as an uncompressed tar containing `raw.npy`,
`tracked.npy`, and a lineage JSON member. It SHALL accept the member named either
`lineages.json` or `lineage.json`, preferring the former, and SHALL raise `ValueError`
when neither is present. A lineage payload that is not a list SHALL be wrapped in one.

#### Scenario: Archive uses the plural member name

- **WHEN** an archive contains `lineages.json`
- **THEN** the lineage data SHALL be read from it

#### Scenario: Archive uses the singular member name

- **WHEN** an archive contains `lineage.json` but not `lineages.json`
- **THEN** the lineage data SHALL be read from `lineage.json`

#### Scenario: Archive carries no lineage member

- **WHEN** an archive contains neither member
- **THEN** the system SHALL raise `ValueError` reporting that no lineage data was found

#### Scenario: Lineage payload is a bare object

- **WHEN** the lineage JSON decodes to a single object rather than a list
- **THEN** it SHALL be wrapped in a one-element list

### Requirement: Labels are relabelled sequentially per movie

The system SHALL relabel each movie's masks so labels are sequential from 1, and SHALL
rewrite that movie's lineage dictionary to the new labels. Keys of the rewritten lineage
SHALL be integers, and each entry's `label` SHALL equal its key.

#### Scenario: Non-sequential labels are compacted

- **WHEN** a movie's masks use labels `[1, 5, 9]`
- **THEN** the relabelled masks SHALL use `[1, 2, 3]` and the lineage SHALL be keyed `1, 2, 3`

#### Scenario: Parent references follow the relabelling

- **WHEN** a cell's `parent` is a label that was remapped
- **THEN** the rewritten entry's `parent` SHALL be the remapped value

#### Scenario: A null parent is preserved

- **WHEN** a cell's `parent` is `null`
- **THEN** the rewritten entry's `parent` SHALL remain `null` rather than becoming an integer

### Requirement: Daughters absent from the mask are dropped with a warning

Relabelling SHALL drop any daughter label that does not appear in the movie's masks, and
SHALL emit a warning naming the cell and the missing daughter. The surviving daughters
SHALL be retained in order.

#### Scenario: A lineage names a daughter with no pixels

- **WHEN** a cell's `daughters` list names a label absent from the mask
- **THEN** that daughter SHALL be omitted from the rewritten entry
- **AND** a warning SHALL be emitted naming the cell and the missing daughter

### Requirement: Frames are recomputed from the masks, not trusted from the lineage

The rewritten lineage's `frames` SHALL be derived by scanning the masks for the frames in
which the label appears, discarding whatever the input lineage recorded.

#### Scenario: The input lineage disagrees with the masks

- **WHEN** an input entry records `frames: [0, 1, 2]` but the label appears only in frames 0 and 2
- **THEN** the rewritten entry's `frames` SHALL be `[0, 2]`

### Requirement: Cell capacity is derived from the busiest lineage

The system SHALL size every per-cell axis of the processed arrays to `max_cells`, the
largest number of lineage entries in any single movie of the split. Movies with fewer
cells SHALL be zero-padded to that width.

#### Scenario: Capacity comes from the largest movie

- **WHEN** a split holds movies with 40, 120 and 331 lineage entries
- **THEN** `max_cells` SHALL be 331 and every movie's arrays SHALL be padded to width 331

### Requirement: Morphology is three unnormalized pixel-space scalars

Per-cell morphology SHALL be `[area, perimeter, eccentricity]` in the units of the
rescaled pixel grid, with no normalization applied. Area and perimeter therefore scale
with the rescaling factor; eccentricity does not.

#### Scenario: Morphology is reported in pixels

- **WHEN** features are extracted for a cell
- **THEN** its morphology SHALL be its region area in pixels, its perimeter in pixels, and its eccentricity

### Requirement: Temporal adjacency encodes lineage as a four-valued matrix

The `labels` array SHALL encode, for each frame transition, `1` on the diagonal for a cell
alive in that frame, `2` from a parent to each daughter at the parent's final frame, `0`
for no relation, and `-1` for padding. Padding SHALL cover both unused cell slots and
frames beyond the longest track.

#### Scenario: A surviving cell is marked on the diagonal

- **WHEN** a cell is present in a frame and has not yet ended
- **THEN** its diagonal entry for that frame SHALL be 1

#### Scenario: A division is marked from parent to daughters

- **WHEN** a cell's final frame is `f` and it has daughters
- **THEN** the entries from that cell to each daughter at frame `f` SHALL be 2

#### Scenario: Unused capacity is marked as padding

- **WHEN** a movie uses fewer than `max_cells` slots
- **THEN** the unused rows and columns SHALL be `-1`

### Requirement: A lone live cell outside the first slot aborts conversion

Temporal adjacency construction SHALL raise `IndexError` when exactly one cell is alive
across a frame transition and that cell occupies a slot other than the first, because the
padding search collapses a single match to one dimension and then indexes it as two. A
lone cell in the first slot is unaffected only because its all-zero index fails the guard
that precedes the indexing. This is divergence 23 and is suspected incorrect.

#### Scenario: A lone live cell outside the first slot

- **WHEN** a movie's lineage leaves exactly one cell alive across a frame transition, in a slot other than the first
- **THEN** conversion SHALL raise `IndexError`

#### Scenario: A lone live cell in the first slot is tolerated

- **WHEN** that single live cell occupies the first slot
- **THEN** conversion SHALL complete, because the all-zero index fails the guard and the padding branch is skipped

### Requirement: `track_length` is stored but never populated

The processed zarr SHALL contain a `track_length` array of shape `(B, N, 2)` and integer
dtype whose every element is zero. No stage of conversion writes to it, and no consumer
reads it. This is recorded as divergence 16: the array is suspected vestigial and is
specified here as current behavior rather than as intended behavior.

#### Scenario: The stored array is entirely zero

- **WHEN** a split is converted and `track_length` is read back
- **THEN** every element SHALL be zero

### Requirement: Spatial adjacency is reconstructed at sampling time, not stored

The dataloader SHALL compute each sample's spatial adjacency from that sample's centroids
when the sample is drawn, rather than reading it from the processed zarr. Conversion
SHALL NOT compute any array it does not return. This narrows divergence 15: the discarded
validity mask is deleted by this change, and the discarded spatial adjacency remains.

#### Scenario: A drawn sample carries adjacency the zarr does not hold

- **WHEN** a sample is drawn from the dataloader
- **THEN** its `adj_matrices` SHALL be derived from its centroids
- **AND** no adjacency SHALL have been read from the processed zarr

#### Scenario: Conversion computes no discarded validity mask

- **WHEN** a split is converted
- **THEN** conversion SHALL NOT build a per-frame validity mask that it does not return

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

### Requirement: Augmentation rigidly transforms appearances and centroids only

When augmentation is enabled, the system SHALL rotate appearance crops and centroids by a
single angle drawn per sample and then translate centroids by a single offset drawn per
sample. Morphologies SHALL NOT be transformed. Centroid rotation SHALL be about the
coordinate origin, not the image centre.

#### Scenario: One angle and one offset apply to the whole sample

- **WHEN** a sample is drawn with augmentation enabled
- **THEN** every frame and every cell in that sample SHALL share the same rotation angle and translation offset

#### Scenario: Morphology is left untransformed

- **WHEN** a sample is augmented
- **THEN** its morphologies SHALL equal the stored values

#### Scenario: Adjacency is unaffected by augmentation

- **WHEN** a sample is augmented
- **THEN** its adjacency SHALL be unchanged, because it is computed before augmentation and the transform is rigid

### Requirement: Backward temporal direction discards augmentation

When the temporal direction is backward, the system SHALL reverse the frame order of
every array and transpose each label matrix, and SHALL rebuild centroids and appearances
from the stored values — discarding any augmentation computed for that sample. This is
recorded as divergence 22: augmentation is silently inert in backward mode.

#### Scenario: Backward samples are unaugmented

- **WHEN** a sample is drawn with augmentation enabled and the direction set to backward
- **THEN** its centroids and appearances SHALL equal the stored values, reversed in time

#### Scenario: Label matrices are transposed

- **WHEN** the direction is backward
- **THEN** each frame's label matrix SHALL be transposed, so parent-to-daughter entries become daughter-to-parent

### Requirement: Validation batches are unaugmented; test batches are not

The validation dataloader SHALL disable augmentation and SHALL NOT shuffle. The training
and test dataloaders SHALL both shuffle and SHALL both leave augmentation at its default
of enabled. The test loader's configuration is recorded as divergence 21: a test split
that is shuffled and augmented cannot produce a stable evaluation.

#### Scenario: Validation is deterministic in content

- **WHEN** validation batches are drawn
- **THEN** they SHALL be unaugmented and in a fixed order

#### Scenario: The test loader shuffles and augments

- **WHEN** a test dataloader is constructed without overriding augmentation
- **THEN** it SHALL shuffle and SHALL augment

### Requirement: Appearance is an aspect-preserving resize to a fixed size

Conversion SHALL extract each cell's appearance by taking its bounding box, padding that
box to a square, masking it to that cell's label, and resizing the result to 32×32
pixels. No cell SHALL be clipped and no aspect ratio SHALL be altered. Conversion SHALL
normalize the crop by the mean and standard deviation of the cell's own non-zero pixels.
This is the same rule inference applies, so a cell presents the same appearance in
training as at inference.

#### Scenario: A large cell is resized rather than clipped

- **WHEN** a cell's extent exceeds 32 pixels on either axis
- **THEN** its stored appearance SHALL contain the whole cell, reduced to fit

#### Scenario: Centroids are stored in image coordinates

- **WHEN** appearance is extracted during conversion
- **THEN** the stored centroids SHALL carry no offset relative to the rescaled image

#### Scenario: Conversion and inference extract appearance identically

- **WHEN** the same frame and mask are passed through conversion and through the
  inference feature path
- **THEN** the resulting appearance crops SHALL be identical
- **AND** the resulting centroids SHALL be identical

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

### Requirement: The conversion output directory parameter is unusable

**Superseded-pending:** a later change replaces this; no characterization test is written
for it. Conversion SHALL accept an output-directory parameter and SHALL raise `NameError`
when any value other than `None` is supplied, because the names it needs are bound only
on the default branch. This is divergence 19. It was previously attributed to
`redesign-feature-extraction`, which does not touch argument binding; the attribution is
corrected here and the fix is not performed.

#### Scenario: Supplying an output directory fails

- **WHEN** conversion is called with an explicit output directory
- **THEN** it SHALL raise `NameError` rather than writing to that directory

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
