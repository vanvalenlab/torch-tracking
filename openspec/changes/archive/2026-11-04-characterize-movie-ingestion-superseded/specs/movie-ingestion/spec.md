# Spec Delta

## Purpose

Turns a caller-supplied time-lapse movie and its segmentation masks into the validated,
rescaled, per-cell feature arrays that the tracking model consumes. This capability owns
everything between the caller's arrays and those features; it never produces masks, and
it stops short of computing embeddings.

## ADDED Requirements

### Requirement: Movie and masks must be rank 4

The system SHALL reject a movie or mask array that is not rank 4, shaped
`(frames, height, width, channels)`, with an error naming the ranks it received.

#### Scenario: A 3-dimensional movie is supplied

- **WHEN** a caller supplies a movie of shape `(T, H, W)` and masks of shape `(T, H, W, 1)`
- **THEN** ingestion SHALL fail with an error reporting that both inputs must be rank 4
- **AND** the error SHALL state the rank actually received for each input

#### Scenario: Both inputs are rank 4

- **WHEN** a caller supplies a movie of shape `(T, H, W, 1)` and masks of shape `(T, H, W, 1)`
- **THEN** ingestion SHALL proceed past rank validation

### Requirement: Movie and masks must agree on every axis except channels

The system SHALL reject inputs whose frame or spatial dimensions differ, reporting both
shapes. The channel axis is exempt, so a multi-channel movie may accompany a
single-channel mask.

#### Scenario: Spatial dimensions disagree

- **WHEN** a caller supplies a movie of shape `(10, 100, 100, 1)` and masks of shape `(10, 200, 200, 1)`
- **THEN** ingestion SHALL fail with an error reporting both shapes

#### Scenario: Only the channel count differs

- **WHEN** a caller supplies a movie of shape `(10, 100, 100, 2)` and masks of shape `(10, 100, 100, 1)`
- **THEN** ingestion SHALL proceed past shape validation

### Requirement: Cell labels are made unique across all frames

The system SHALL relabel every mask so that no label is reused in two frames, because
downstream tracking treats a repeated label as a collision. Within a frame, labels SHALL
be assigned in ascending order of the original label.

#### Scenario: The same label appears in consecutive frames

- **WHEN** masks label two unrelated cells `1` in frame 0 and frame 1
- **THEN** after ingestion those two cells SHALL carry different labels
- **AND** no label SHALL appear in more than one frame

#### Scenario: Background is preserved

- **WHEN** masks contain background pixels valued `0`
- **THEN** those pixels SHALL remain `0` after relabelling

### Requirement: Masks at the caller's original resolution are retained

The system SHALL retain a copy of the relabelled masks at the resolution the caller
supplied, taken before any rescaling, so that results can be reported in the caller's
coordinate space rather than the model's.

#### Scenario: Ingestion rescales the movie

- **WHEN** a caller supplies masks of shape `(T, H, W, 1)` with a resolution requiring rescaling
- **THEN** the retained original-resolution masks SHALL still have spatial dimensions `H` and `W`

### Requirement: Input is rescaled to the model's native resolution

The system SHALL rescale the movie and masks so that one output pixel spans the
resolution the model was trained at, 0.55 µm. The scale factor SHALL be
`mpp / 0.55`, where `mpp` is the caller's declared microns per pixel. Output dimensions
SHALL be truncated to integers. Masks SHALL be resampled without interpolating between
labels.

#### Scenario: Caller's resolution is coarser than the model's

- **WHEN** a caller declares `mpp = 0.65` for a movie of shape `(T, 100, 100, 1)`
- **THEN** the rescaled movie SHALL have spatial dimensions `(118, 118)`, enlarging the image so each pixel spans 0.55 µm

#### Scenario: Caller's resolution already matches the model's

- **WHEN** a caller declares `mpp = 0.55`
- **THEN** the scale factor SHALL be 1.0 and spatial dimensions SHALL be unchanged

#### Scenario: Masks are resampled without inventing labels

- **WHEN** masks are rescaled
- **THEN** every label in the rescaled masks SHALL be a label that was present before rescaling

### Requirement: Each cell yields appearance, morphology and centroid features

For every labelled cell in every frame, the system SHALL produce a fixed-size appearance
crop centred on the cell, a morphology vector of area, perimeter and eccentricity in that
order, and a centroid. Features SHALL be ordered by ascending cell label within a frame.

#### Scenario: A frame containing three cells

- **WHEN** a frame contains three labelled cells
- **THEN** three appearance crops, three morphology vectors and three centroids SHALL be produced
- **AND** they SHALL be ordered by ascending cell label

#### Scenario: Morphology component order

- **WHEN** morphology is produced for a cell
- **THEN** its components SHALL be area, perimeter and eccentricity, in that order

#### Scenario: Appearance crops are masked to their own cell

- **WHEN** a neighbouring cell overlaps a cell's appearance crop window
- **THEN** pixels belonging to the neighbour SHALL be zero in that crop

### Requirement: Feature arrays are padded to the busiest frame

The system SHALL size per-frame feature arrays to the largest number of cells found in
any single frame and SHALL zero-pad frames containing fewer, so that every frame presents
the same shape to the model.

#### Scenario: Frames contain differing cell counts

- **WHEN** one frame contains 5 cells and another contains 3
- **THEN** both frames' feature arrays SHALL have a cell dimension of 5
- **AND** the 2 unused entries in the second frame SHALL be zero

### Requirement: Cells are connected to neighbours within a distance threshold

The system SHALL build a per-frame adjacency graph connecting two cells when the
Euclidean distance between their centroids is greater than zero and less than the
configured threshold, which defaults to 64 pixels. Padding entries SHALL never be
connected to anything.

#### Scenario: Two cells closer than the threshold

- **WHEN** two cells in a frame have centroids 10 pixels apart and the threshold is 64
- **THEN** they SHALL be connected in that frame's adjacency graph

#### Scenario: Two cells beyond the threshold

- **WHEN** two cells have centroids 200 pixels apart and the threshold is 64
- **THEN** they SHALL NOT be connected

#### Scenario: A cell is not connected to itself

- **WHEN** adjacency is built for any frame
- **THEN** no cell SHALL be connected to itself

#### Scenario: Padding entries are isolated

- **WHEN** a frame contains fewer cells than the padded array allows
- **THEN** every padding entry SHALL be disconnected from every other entry

### Requirement: Ingestion is deterministic

Given identical inputs and configuration, the system SHALL produce bit-identical feature
and adjacency arrays on every run and on every supported device. No step SHALL depend on
random state.

#### Scenario: The same movie is ingested twice

- **WHEN** the same movie, masks and configuration are ingested twice in one session
- **THEN** the resulting appearance, morphology, centroid and adjacency arrays SHALL be identical

### Requirement: Centroids are reported in padded-crop coordinates

The system reports centroids offset by the appearance crop size, 32 pixels, on both
spatial axes, because they are measured after an internal padding step. Consumers
comparing a centroid against an image coordinate MUST subtract that offset.

This behavior is **suspected incorrect** and is recorded, not endorsed. It is harmless
only because every consumer uses centroid *differences*, which cancel the offset.

#### Scenario: A cell at a known image position

- **WHEN** a frame contains a single cell centred at image coordinate `(19.5, 19.5)`
- **THEN** its reported centroid SHALL be `(51.5, 51.5)`

#### Scenario: Pairwise distances are unaffected

- **WHEN** adjacency is computed from offset centroids
- **THEN** the distances SHALL equal those computed from unoffset centroids

### Requirement: A mask frame without background pixels fails

The system SHALL raise `IndexError` when a mask frame contains no pixel valued `0`,
because the cell count is derived by subtracting one from the number of distinct values,
assuming background is present.

This behavior is **suspected incorrect** and is recorded, not endorsed. A fully tiled
mask is a legitimate input that a caller cannot currently supply.

#### Scenario: Cells cover the entire frame

- **WHEN** a mask frame is composed entirely of labels `1` and `2` with no background
- **THEN** ingestion SHALL raise `IndexError`

#### Scenario: A single background pixel is enough

- **WHEN** the same frame has one pixel set to `0`
- **THEN** ingestion SHALL succeed and SHALL report both cells

### Requirement: An all-zero appearance crop is left unnormalized

When a cell's appearance crop contains no non-zero pixels, the system SHALL emit
`RuntimeWarning: Mean of empty slice` and SHALL leave the crop as zeros rather than
normalizing it.

The guard intended to prevent this is **suspected incorrect** and is recorded, not
endorsed: it tests the arity of an index tuple rather than whether it is empty, so it
never prevents anything. The crop escapes corruption only because assigning through an
empty index does nothing.

#### Scenario: A cell sits on an all-zero image region

- **WHEN** a labelled cell covers a region where the movie is entirely zero
- **THEN** a `RuntimeWarning` SHALL be emitted
- **AND** the resulting appearance crop SHALL be all zeros rather than NaN

### Requirement: The rescaling convention is documented as disagreeing with training

Ingestion SHALL scale by `mpp / 0.55`, which matches the physics: data at a coarser
resolution must be enlarged to put 0.55 µm in each pixel. The training-data pipeline
scales by the reciprocal, `0.55 / mpp`.

This disagreement is recorded here so the inference contract is unambiguous. Resolving it
belongs to `training-data`.

#### Scenario: A movie coarser than the model's resolution

- **WHEN** a movie at `mpp = 0.65` is ingested for inference
- **THEN** it SHALL be enlarged by a factor of `0.65 / 0.55`, not reduced by `0.55 / 0.65`

#### Scenario: The default conceals the disagreement

- **WHEN** a caller does not declare `mpp`
- **THEN** the default of 0.55 SHALL yield a scale factor of 1.0, performing no rescaling in either convention
