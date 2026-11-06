# Spec Delta

## ADDED Requirements

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

## MODIFIED Requirements

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

## REMOVED Requirements

### Requirement: A lone live cell outside the first slot aborts conversion

**Reason**: This is divergence 23, and it is **Fixed** rather than moot: the whole-movie
padding search that caused it is replaced by a rule decided deliberately, not merely
deleted. The search collapsed a single match to one dimension and then indexed it as two,
raising `IndexError`; a lone cell in the first slot escaped only because its all-zero
index failed the preceding guard, and therefore received no padding at all. Both halves
are closed by "Padding marks every slot carrying no cell in that frame", which is
evaluated per slot and has no search to collapse.

**Migration**: Every processed zarr must be regenerated, because temporal adjacency is no
longer stored at all. Weights trained against the old targets are superseded: the padding
rule changed, so the loss was computed over a different set of entries. The
characterization test pinning this behavior is deleted in this change.

### Requirement: `track_length` is stored but never populated

**Reason**: This is divergence 16. The `(B, N, 2)` integer array was allocated, returned
and written into every processed zarr, and no stage of conversion ever assigned to it;
nothing read it. It is removed rather than populated. The per-cell first and last frame
its shape suggests is already carried by the lineage, which now travels in the zarr, so
nothing replaces it. Replaced by "The processed zarr holds exactly four arrays". The
unrelated *window* parameter of the same name is untouched.

**Migration**: Any consumer reading the array by name must stop, though none exists. The
characterization test pinning this behavior is deleted in this change, per the project's
standing rule that a change fixing a pinned divergence deletes the test that pinned it.

### Requirement: Backward temporal direction discards augmentation

**Reason**: This is divergence 22, recorded **Moot** rather than Fixed. Backward temporal
direction was an experiment that was never removed; nothing in the repository selected
it. Rather than decide what a time-reversed division should teach the model, the feature
is deleted. Following the precedent of divergence 11, the code holding the defect is
removed rather than analyzed and corrected.

**Migration**: Any caller passing a temporal direction must drop the argument; forward is
the only direction and is no longer selectable. No training run, evaluation or published
figure used backward mode, so no result changes. Both characterization tests covering
backward behavior are deleted in this change.

### Requirement: The processed zarr holds exactly six arrays

**Reason**: Three arrays leave at once — the retired `track_length`, and the temporal
adjacency that is now built per window instead of stored. Replaced by "The processed zarr
holds exactly four arrays".

**Migration**: Regeneration is required, not optional. Unlike the five-to-six change that
added `mpp`, a pre-change zarr cannot drive the new dataloader at all: it carries no
lineage, so no window can be built from it. Reconversion takes roughly 7.5 minutes for
all three splits and reclaims about 3.9 GiB on the train split.

### Requirement: The conversion output directory parameter is unusable

**Reason**: This is divergence 19. Conversion accepted an output-directory argument and
raised `NameError` for any value but `None`, because the names it needed were bound only
on the default branch. Replaced by "Conversion writes to an output directory it is
given".

**Migration**: Callers that worked around the defect by converting in place and moving the
outputs afterwards may now pass the directory directly. No existing call breaks: the
default behavior, writing beside the `.trks` file, is unchanged. No characterization test
pinned this requirement, which was marked superseded-pending, so none is deleted.

### Requirement: Per-movie resolution comes from an undocumented sidecar

**Reason**: This is divergence 20. Conversion required `data-source.npz` beside the
`.trks` file, keyed by split name, and failed without it, while no document said so.
Replaced by "Per-movie resolution is supplied directly or read from a documented
sidecar", which writes the layout down and allows the caller to bypass it.

**Migration**: Existing callers are unaffected — the sidecar remains the fallback and is
read exactly as before. Callers holding resolutions already may now pass them and drop
the file. Conversion still fails when neither is available; a default resolution is
deliberately not offered, because silently mis-scaling every movie is the failure
divergence 12 already cost a retrain.

### Requirement: Validation batches are unaugmented; test batches are not

**Reason**: This is divergence 21. The test dataloader shuffled and augmented, so a split
intended for evaluation was served in random order with random rotations applied. Nothing
outside the dataloader module constructed a test loader — evaluation reads the zarr
directly — so no published figure was produced through it. Replaced by "Validation and
test batches are both unaugmented and ordered".

**Migration**: No result changes, because nothing used the test loader. Any future caller
wanting shuffled or augmented samples from the test split must ask for them explicitly.
The characterization test pinning this behavior is deleted in this change.
