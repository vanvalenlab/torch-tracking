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
shapes. The channel axis is exempt from that comparison, so a multi-channel movie
accompanying a single-channel mask SHALL pass validation. Whether the model accepts the
resulting channel count is a separate contract owned by `tracking-model`.

#### Scenario: Spatial dimensions disagree

- **WHEN** a caller supplies a movie of shape `(10, 100, 100, 1)` and masks of shape `(10, 200, 200, 1)`
- **THEN** ingestion SHALL fail with an error reporting both shapes

#### Scenario: Only the channel count differs

- **WHEN** a caller supplies a movie of shape `(10, 100, 100, 2)` and masks of shape `(10, 100, 100, 1)`
- **THEN** shape validation SHALL NOT reject the input
- **AND** any later failure SHALL come from a different contract, not from this comparison

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

### Requirement: A mask frame needs no background pixels

The system SHALL count a frame's cells by counting its distinct non-zero labels, so that a
fully tiled field of view — a mask with no pixel valued `0` — is ingested like any other.
Presence of background SHALL NOT be assumed.

#### Scenario: Cells cover the entire frame

- **WHEN** a mask frame is composed entirely of labels `1` and `2` with no background
- **THEN** ingestion SHALL succeed and SHALL report both cells

#### Scenario: A frame with background is unaffected

- **WHEN** the same frame has one pixel set to `0`
- **THEN** ingestion SHALL succeed and SHALL still report exactly two cells

### Requirement: Masks at the caller's original resolution are retained

The system SHALL retain a copy of the relabelled masks at the resolution the caller
supplied, taken before any rescaling, so that results can be reported in the caller's
coordinate space rather than the model's.

#### Scenario: Ingestion rescales the movie

- **WHEN** a caller supplies masks of shape `(T, H, W, 1)` with a resolution requiring rescaling
- **THEN** the retained original-resolution masks SHALL still have spatial dimensions `H` and `W`

### Requirement: Input is rescaled to the model's native resolution

The system SHALL rescale the movie and masks so that one output pixel spans the resolution
the model was trained at. The scale factor SHALL be `mpp / MODEL_MPP`, where `mpp` is the
caller's declared microns per pixel and `MODEL_MPP` is a tuned constant, currently 0.55 µm.
Output dimensions SHALL be truncated to integers. Masks SHALL be resampled without
interpolating between labels.

#### Scenario: Caller's resolution is coarser than the model's

- **WHEN** a caller declares `mpp = 0.65` for a movie of shape `(T, 100, 100, 1)`
- **THEN** the rescaled movie SHALL have spatial dimensions `(118, 118)`, enlarging the image so each pixel spans the model resolution

#### Scenario: Caller's resolution already matches the model's

- **WHEN** a caller declares `mpp` equal to `MODEL_MPP`
- **THEN** the scale factor SHALL be 1.0 and spatial dimensions SHALL be unchanged

#### Scenario: Masks are resampled without inventing labels

- **WHEN** masks are rescaled
- **THEN** every label in the rescaled masks SHALL be a label that was present before rescaling

#### Scenario: Coarser input is enlarged, not reduced

- **WHEN** a movie coarser than `MODEL_MPP` is ingested
- **THEN** it SHALL be enlarged, because putting a fixed physical distance in each pixel requires more pixels, not fewer

### Requirement: Each cell yields appearance, morphology and centroid features

For every labelled cell in every frame, the system SHALL produce an appearance crop
centred on the cell, a morphology vector of area, perimeter and eccentricity in that
order, and a centroid. Features SHALL be ordered by ascending cell label within a frame.
In fixed-crop mode, masking a crop to its own cell is governed by the normalization flag;
resize mode masks unconditionally.

#### Scenario: A frame containing three cells

- **WHEN** a frame contains three labelled cells
- **THEN** three appearance crops, three morphology vectors and three centroids SHALL be produced
- **AND** they SHALL be ordered by ascending cell label

#### Scenario: Morphology component order

- **WHEN** morphology is produced for a cell
- **THEN** its components SHALL be area, perimeter and eccentricity, in that order

#### Scenario: Appearance crops are masked to their own cell

- **WHEN** a neighbouring cell overlaps a cell's appearance crop window and normalization is enabled
- **THEN** pixels belonging to the neighbour SHALL be zero in that crop

#### Scenario: Disabling normalization also disables masking in fixed-crop mode

- **WHEN** appearance is extracted in fixed-crop mode with normalization disabled
- **THEN** a neighbour overlapping the window SHALL remain visible in the crop, because
  the masking step shares the normalization flag
- **AND** the same request in resize mode SHALL still mask the neighbour, because that
  branch masks unconditionally

### Requirement: Feature arrays are padded to the busiest frame

The system SHALL size per-frame feature arrays to the largest number of cells found in
any single frame and SHALL zero-pad frames containing fewer, so that every frame presents
the same shape to the model.

#### Scenario: Frames contain differing cell counts

- **WHEN** one frame contains 5 cells and another contains 3
- **THEN** both frames' feature arrays SHALL have a cell dimension of 5
- **AND** the 2 unused entries in the second frame SHALL be zero

### Requirement: Cells are connected to neighbours within a distance threshold

The system SHALL build a per-frame adjacency graph connecting two cells when the Euclidean
distance between their centroids is greater than zero and less than `distance_threshold`,
a tuned constant currently defaulting to 64 pixels. Padding entries SHALL never be
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

### Requirement: Appearance is a fixed-size crop that clips large cells

**Pending-fix:** `correct-scale-convention` replaces this with an aspect-preserving
resize. The system SHALL extract each cell's appearance as a fixed 32×32 pixel window
centred on its centroid, taken from a zero-padded image. A cell larger than the window
SHALL be clipped rather than resized, discarding its periphery.

#### Scenario: A large cell is clipped rather than resized

- **WHEN** a cell's extent exceeds 32 pixels on either axis
- **THEN** its appearance SHALL contain only the central 32×32 pixels of the cell

#### Scenario: A small cell is padded, not magnified

- **WHEN** a cell is smaller than the window
- **THEN** the remainder of its crop SHALL be zero rather than an enlargement of the cell

### Requirement: Centroids are reported in padded-crop coordinates

**Pending-fix:** `correct-scale-convention` makes this moot. Centroids SHALL be reported
offset by the appearance crop size, 32 pixels, on both spatial axes, because they are
measured after an internal padding step. A consumer comparing a centroid against an image
coordinate MUST subtract it. This is divergence 11 and is suspected incorrect.

#### Scenario: A cell at a known image position

- **WHEN** a frame contains a single cell centred at image coordinate `(19.5, 19.5)`
- **THEN** its reported centroid SHALL be `(51.5, 51.5)`

#### Scenario: Pairwise distances are unaffected

- **WHEN** adjacency is computed from offset centroids
- **THEN** the distances SHALL equal those computed from unoffset centroids
- **AND** the offset SHALL therefore be harmless to every consumer that uses centroid
  differences rather than absolute positions

### Requirement: An all-zero appearance crop is left unnormalized

**Pending-fix:** `correct-scale-convention` makes this moot by replacing the fixed-crop
branch it lives in. When a cell's appearance crop contains no non-zero pixels, the system
SHALL emit `RuntimeWarning: Mean of empty slice` and SHALL leave the crop as zeros rather
than normalizing it. The guard intended to prevent this tests the arity of an index tuple
rather than whether it is empty, so it never prevents anything. This is divergence 14.

#### Scenario: A cell sits on an all-zero image region

- **WHEN** a labelled cell covers a region where the movie is entirely zero
- **THEN** a `RuntimeWarning` SHALL be emitted
- **AND** the resulting appearance crop SHALL be all zeros rather than NaN
