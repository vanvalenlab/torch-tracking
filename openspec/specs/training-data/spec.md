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
SHALL be integers, and each entry's `label` SHALL equal its key. The input lineage SHALL
be accepted keyed by integers or by the strings a JSON document yields, so a lineage this
package produced can be relabelled again. This closes the input half of divergence 1.

#### Scenario: Non-sequential labels are compacted

- **WHEN** a movie's masks use labels `[1, 5, 9]`
- **THEN** the relabelled masks SHALL use `[1, 2, 3]` and the lineage SHALL be keyed `1, 2, 3`

#### Scenario: Parent references follow the relabelling

- **WHEN** a cell's `parent` is a label that was remapped
- **THEN** the rewritten entry's `parent` SHALL be the remapped value

#### Scenario: A null parent is preserved

- **WHEN** a cell's `parent` is `null`
- **THEN** the rewritten entry's `parent` SHALL remain `null` rather than becoming an integer

#### Scenario: A lineage read from JSON has string keys

- **WHEN** a lineage decoded from a `.trk` archive is keyed by strings
- **THEN** relabelling SHALL resolve every entry
- **AND** the rewritten lineage SHALL be keyed by integers

#### Scenario: A lineage already keyed by integers is relabelled

- **WHEN** a lineage keyed by integers is passed to relabelling
- **THEN** it SHALL be resolved without error
- **AND** the rewritten lineage SHALL be keyed by integers

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
present in both frames, `2` from a parent to each daughter at the parent's final frame, `0`
for two present cells that are not the same, and `-1` for padding. Padding SHALL mark every
entry with a slot carrying no cell, as the requirement of that name states; it no longer
covers whole frames, because windows are built individually and never extend past a
movie's extent.

#### Scenario: A surviving cell is marked on the diagonal

- **WHEN** a cell is present in a frame and in the next
- **THEN** its diagonal entry for that transition SHALL be 1

#### Scenario: A division is marked from parent to daughters

- **WHEN** a cell's final frame is `f` and it has daughters
- **THEN** the entries from that cell to each daughter at frame `f` SHALL be 2

#### Scenario: Unused capacity is marked as padding

- **WHEN** a movie uses fewer than `max_cells` slots
- **THEN** the unused rows and columns SHALL be `-1`

#### Scenario: A cell alive in its final frame keeps its diagonal

- **WHEN** a cell is present in a frame and in the next, ends at that next frame, and holds
  the highest slot
- **THEN** its diagonal entry for that transition SHALL be 1
- **AND** SHALL NOT be overwritten by padding, because the cell is present in both frames

### Requirement: Spatial adjacency is reconstructed at sampling time, not stored

The dataloader SHALL compute each sample's spatial adjacency from that sample's centroids
when the sample is drawn, rather than reading it from the processed zarr. Conversion
SHALL NOT compute any array it does not return. This closes divergence 15: the discarded
validity mask was deleted by `redesign-feature-extraction`, and the discarded spatial
adjacency is deleted here.

#### Scenario: A drawn sample carries adjacency the zarr does not hold

- **WHEN** a sample is drawn from the dataloader
- **THEN** its `adj_matrices` SHALL be derived from its centroids
- **AND** no adjacency SHALL have been read from the processed zarr

#### Scenario: Conversion computes no discarded validity mask

- **WHEN** a split is converted
- **THEN** conversion SHALL NOT build a per-frame validity mask that it does not return

#### Scenario: Conversion computes no discarded spatial adjacency

- **WHEN** a split is converted
- **THEN** conversion SHALL NOT compute a spatial adjacency for any frame
- **AND** SHALL NOT allocate an array to hold one
- **AND** SHALL NOT require a distance threshold, because conversion builds no graph

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
coordinate origin, not the image centre. The angle and offset drawn for a sample SHALL NOT
outlive that sample's transformation.

#### Scenario: One angle and one offset apply to the whole sample

- **WHEN** a sample is drawn with augmentation enabled
- **THEN** every frame and every cell in that sample SHALL share the same rotation angle and translation offset

#### Scenario: Morphology is left untransformed

- **WHEN** a sample is augmented
- **THEN** its morphologies SHALL equal the stored values

#### Scenario: Adjacency is unaffected by augmentation

- **WHEN** a sample is augmented
- **THEN** its adjacency SHALL be unchanged, because it is computed before augmentation and the transform is rigid

#### Scenario: A sample's draw is not stored on the dataset

- **WHEN** a sample is augmented
- **THEN** the angle and offset drawn for it SHALL NOT be recorded on the dataset object
- **AND** no later sample's transformation SHALL be able to observe them

#### Scenario: Concurrent sampling cannot interleave two draws

- **WHEN** two samples are drawn from one dataset object at the same time
- **THEN** each SHALL be transformed by its own angle and offset
- **AND** the result SHALL NOT depend on the order in which the two draws complete

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

### Requirement: Temporal adjacency is built per window when the sample is drawn

The dataloader SHALL build each sample's temporal adjacency from the movie's lineage when
the sample is drawn, rather than reading it from the processed zarr, and conversion SHALL
NOT compute or store one. Because window generation never emits a window extending past a
movie's occupied extent, no frame-axis padding is produced. This closes divergences 23
and 38.

#### Scenario: A drawn sample carries targets the zarr does not hold

- **WHEN** a sample is drawn from the dataloader
- **THEN** its `labels` SHALL be derived from the movie's lineage
- **AND** no temporal adjacency SHALL have been read from the processed zarr

#### Scenario: Conversion stores no temporal adjacency

- **WHEN** a split is converted
- **THEN** the processed zarr SHALL contain no temporal adjacency array
- **AND** conversion SHALL NOT build a whole-movie adjacency at any point

#### Scenario: No frame-axis padding is produced

- **WHEN** a window is built for any movie
- **THEN** every frame transition in that window SHALL lie within the movie's occupied extent
- **AND** no transition SHALL be padded for being beyond the longest track

#### Scenario: A lone live cell is handled like any other

- **WHEN** a movie's lineage leaves exactly one cell alive across a frame transition
- **THEN** the window SHALL be built without error, whatever slot that cell occupies
- **AND** its padding SHALL follow the same rule as any other window

### Requirement: Padding marks every slot carrying no cell in that frame

A temporal adjacency entry compares a cell of one frame (its row) with a cell of the next
(its column), and SHALL be `-1` unless both slots carry a cell. A slot beyond the movie's
cell count and a real cell absent from the frame it is read in are indistinguishable in the
features, so the targets SHALL NOT distinguish them either.

#### Scenario: A slot beyond the movie's cell count is padding

- **WHEN** a movie uses fewer slots than `max_cells`
- **THEN** the unused rows and columns SHALL be `-1`

#### Scenario: A real cell absent from a frame is padding there

- **WHEN** a slot belongs to a cell of this movie that is not present in the frame its row
  or its column is read in, whether it has not yet been born or has already ended
- **THEN** that row or column SHALL be `-1` for that frame transition
- **AND** SHALL NOT be `0`, because the features for that slot are indistinguishable from
  a slot beyond capacity

#### Scenario: A newborn cell is graded as none of the existing tracks

- **WHEN** a cell's first frame is `f + 1`
- **THEN** at the transition from `f`, its column SHALL be `0` against every track present
  in frame `f`, and its row SHALL be `-1`

#### Scenario: A cell that ends is graded as matching no candidate

- **WHEN** a cell's final frame is `f`, it has no daughters, and other cells are present
  in frame `f + 1`
- **THEN** at the transition from `f`, its row SHALL be `0` against each of them
- **AND** its own diagonal entry SHALL be `-1`, because no cell occupies its column in
  frame `f + 1`

#### Scenario: A cell missing from the next frame is not graded as the same cell

- **WHEN** a cell is present in frame `f` and absent from frame `f + 1`, although it
  returns later
- **THEN** its diagonal entry at the transition from `f` SHALL be `-1`, not `1`

#### Scenario: A dividing parent keeps its daughter entries

- **WHEN** a cell's final frame is `f` and it has daughters
- **THEN** the entries from that cell to each daughter at frame `f` SHALL be `2`
- **AND** the padding rule SHALL NOT erase them, including where a daughter's first frame
  is not `f + 1`

#### Scenario: A pair of present cells is never padding

- **WHEN** one cell is present in frame `f` and another in frame `f + 1`
- **THEN** the entry between them at the transition from `f` SHALL NOT be `-1`

### Requirement: The processed zarr carries the corrected lineage

Conversion SHALL store each movie's corrected lineage in the processed zarr, so the
dataloader can build temporal adjacency without reading any sibling file. A processed
zarr that does not carry it SHALL be rejected with an error naming what is missing,
rather than falling back to a sibling file.

#### Scenario: The lineage travels with the arrays

- **WHEN** a split is converted
- **THEN** the processed zarr SHALL carry the corrected lineage for every movie
- **AND** the dataloader SHALL build windows from it without opening another file

#### Scenario: A zarr without the lineage is rejected

- **WHEN** a processed zarr written before this requirement is opened by the dataloader
- **THEN** it SHALL fail with an error naming the missing lineage
- **AND** SHALL NOT fall back to a sibling `<split>.json`

### Requirement: The processed zarr holds exactly four arrays

Conversion SHALL write a processed zarr containing `appearances`, `morphologies`,
`centroids` and `mpp`, and nothing else. Shapes SHALL be `(B, T, N, 32, 32, 1)`,
`(B, T, N, 3)`, `(B, T, N, 2)` and `(B,)` respectively, where `N` is `max_cells`. This
closes divergences 15 and 16.

#### Scenario: Converting a split writes the documented layout

- **WHEN** a `.trks` split is converted
- **THEN** the processed zarr SHALL hold those four arrays at those shapes
- **AND** SHALL contain no spatial adjacency array
- **AND** SHALL contain no `track_length` array
- **AND** SHALL contain no temporal adjacency array

#### Scenario: An unprocessed zarr and a lineage JSON are written alongside

- **WHEN** a `.trks` split is converted
- **THEN** a sibling `<split>.zarr` holding `X`, `y` and `mpp` SHALL be written
- **AND** a sibling `<split>.json` holding the corrected lineages SHALL be written

### Requirement: The dataloader reads only the arrays it uses

The dataloader SHALL read `appearances`, `morphologies`, `centroids` and `mpp` by name,
and SHALL NOT load every array a processed zarr happens to hold. A zarr carrying a
retired array SHALL therefore cost no memory for data no stage reads.

#### Scenario: Retired arrays are not loaded

- **WHEN** a processed zarr carrying a retired array is opened
- **THEN** the dataloader SHALL NOT read that array into memory

#### Scenario: A missing required array is an error

- **WHEN** a processed zarr lacks one of the four arrays
- **THEN** the dataloader SHALL fail with an error naming the missing array

### Requirement: Conversion writes to an output directory it is given

Conversion SHALL write its three outputs — the unprocessed zarr, the processed zarr and
the corrected lineage JSON — into a caller-supplied output directory when one is given,
and beside the `.trks` file otherwise. The three SHALL NOT be separable: all of them
follow the same choice. This closes divergence 19.

#### Scenario: An explicit output directory is honoured

- **WHEN** conversion is called with an output directory
- **THEN** `<split>.zarr`, `<split>_proc.zarr` and `<split>.json` SHALL all be written
  under that directory
- **AND** no output SHALL be written beside the `.trks` file

#### Scenario: The default remains beside the archive

- **WHEN** conversion is called without an output directory
- **THEN** all three outputs SHALL be written beside the `.trks` file, as before

#### Scenario: A missing output directory is an error, not a silent fallback

- **WHEN** conversion is given an output directory that cannot be written
- **THEN** it SHALL fail naming that directory
- **AND** SHALL NOT fall back to writing beside the `.trks` file

### Requirement: Per-movie resolution is supplied directly or read from a documented sidecar

Conversion SHALL obtain each movie's declared resolution from resolutions supplied by the
caller, or, when none are supplied, from a `data-source.npz` sidecar beside the `.trks`
file, keyed by split name, taking the third column of that split's metadata array. That
layout SHALL be documented. Conversion SHALL fail when neither source is available. This
closes divergence 20.

#### Scenario: Resolutions supplied by the caller are used

- **WHEN** conversion is given one resolution per movie
- **THEN** those values SHALL be used
- **AND** `data-source.npz` SHALL NOT be read

#### Scenario: The sidecar is the fallback, and its layout is documented

- **WHEN** conversion is called without resolutions and a `data-source.npz` sits beside
  the `.trks` file
- **THEN** each movie's resolution SHALL be read from the third column of that split's
  metadata array
- **AND** the required file name, key and column SHALL be stated in the project
  documentation

#### Scenario: Neither source available is a failure

- **WHEN** conversion is called without resolutions and no `data-source.npz` is present
- **THEN** conversion SHALL fail naming what is missing
- **AND** SHALL NOT fall back to a default resolution

#### Scenario: Supplied resolutions must cover every movie

- **WHEN** the caller supplies fewer resolutions than the split holds movies
- **THEN** conversion SHALL fail rather than recycle or pad the values

### Requirement: Validation and test batches are both unaugmented and ordered

The validation and test dataloaders SHALL both disable augmentation and SHALL NOT
shuffle, so either split can produce a stable, repeatable evaluation. The training
dataloader SHALL shuffle and SHALL leave augmentation enabled by default. This closes
divergence 21.

#### Scenario: Test batches are deterministic in content and order

- **WHEN** test batches are drawn
- **THEN** they SHALL be unaugmented
- **AND** they SHALL be in a fixed order

#### Scenario: Validation is unchanged

- **WHEN** validation batches are drawn
- **THEN** they SHALL be unaugmented and in a fixed order

#### Scenario: Training still shuffles and augments

- **WHEN** a training dataloader is constructed without overriding augmentation
- **THEN** it SHALL shuffle and SHALL augment

#### Scenario: Two passes over the test split agree

- **WHEN** the test dataloader is iterated twice
- **THEN** the two passes SHALL yield the same samples in the same order with the same
  values

### Requirement: The data modules carry no entry point nothing runs

A module in the data path SHALL NOT carry a script block that nothing invokes. Such a block
is not exercised, so it goes stale without anyone noticing and then states, in code, a
configuration that contradicts the one the package actually uses. Where a block held checks
worth keeping, those checks SHALL move into the test suite rather than be repaired in
place. This closes divergence 65 and the equivalent block in the conversion module.

#### Scenario: Importing a data module runs nothing

- **WHEN** a data-path module is imported
- **THEN** no data SHALL be read, converted or written
- **AND** no loader, device or dataset path SHALL be constructed

#### Scenario: No module states a second configuration

- **WHEN** the package is searched for training or conversion settings
- **THEN** no module SHALL declare a device, dataset location or configuration key outside
  the documented configuration surface

#### Scenario: A removed block leaves a record of where its checks went

- **WHEN** a script block is removed from a module
- **THEN** the module SHALL note where the checks it held now run, or that it held none
- **AND** the removal SHALL be recorded in the divergence register

#### Scenario: Real command-line entry points are unaffected

- **WHEN** a module exposes a documented command line that callers invoke
- **THEN** that entry point SHALL remain
