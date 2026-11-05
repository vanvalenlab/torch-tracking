# movie-ingestion Specification

## Purpose

Turns a caller-supplied time-lapse movie and its segmentation masks into the validated,
rescaled, per-cell feature arrays that the tracking model consumes. This capability owns
everything between the caller's arrays and those features; it never produces masks, and
it stops short of computing embeddings.

## Requirements

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

### Requirement: Every cell yields appearance, morphology and centroid features

For every labelled cell in every frame, the system SHALL produce an appearance crop, a
morphology vector of area, perimeter and eccentricity in that order, and a centroid.
Features SHALL be ordered by ascending cell label within a frame. Every appearance crop
SHALL be masked to its own cell unconditionally.

#### Scenario: A frame containing three cells

- **WHEN** a frame contains three labelled cells
- **THEN** three appearance crops, three morphology vectors and three centroids SHALL be produced
- **AND** they SHALL be ordered by ascending cell label

#### Scenario: Morphology component order

- **WHEN** morphology is produced for a cell
- **THEN** its components SHALL be area, perimeter and eccentricity, in that order

#### Scenario: Appearance crops are masked to their own cell

- **WHEN** a neighbouring cell overlaps a cell's bounding box
- **THEN** pixels belonging to the neighbour SHALL be zero in that crop

#### Scenario: Masking cannot be switched off

- **WHEN** appearance is extracted by any caller
- **THEN** masking SHALL be applied
- **AND** no parameter SHALL exist that selects a crop mode or disables masking or
  normalization

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
random state. The per-cell resampling introduced by the appearance resize SHALL be
deterministic.

#### Scenario: The same movie is ingested twice

- **WHEN** the same movie, masks and configuration are ingested twice in one session
- **THEN** the resulting appearance, morphology, centroid and adjacency arrays SHALL be identical

#### Scenario: Resampling does not introduce variation

- **WHEN** the same cell is resized to the appearance crop size twice
- **THEN** both crops SHALL be bit-identical

### Requirement: Appearance is an aspect-preserving resize to a fixed size

The system SHALL extract each cell's appearance by taking its bounding box, padding that
box to a square, masking it to that cell's own label, and resizing the result to 32×32
pixels. No cell SHALL be clipped and no cell's aspect ratio SHALL be altered. Appearance
carries texture, shape and contrast, and SHALL NOT carry physical size, which is the
morphology vector's responsibility.

#### Scenario: A large cell is resized rather than clipped

- **WHEN** a cell's extent exceeds 32 pixels on either axis
- **THEN** its appearance SHALL contain the whole cell, reduced to fit
- **AND** no part of the cell SHALL be discarded

#### Scenario: A small cell is magnified, not padded

- **WHEN** a cell is smaller than 32 pixels on both axes
- **THEN** its appearance SHALL be the cell enlarged to fill the crop
- **AND** the crop SHALL NOT be the cell surrounded by zero padding

#### Scenario: An elongated cell keeps its proportions

- **WHEN** a cell's bounding box is 20 pixels by 60 pixels
- **THEN** that box SHALL first be padded to 60 by 60 about the cell
- **AND** the resized appearance SHALL show the cell three times longer than it is wide,
  not squared off to fill the crop

#### Scenario: Two cells of identical shape and differing size look the same

- **WHEN** two cells have the same shape, brightness and texture but one is twice the
  linear size of the other
- **THEN** their appearance crops SHALL be equal to within resampling error
- **AND** their morphology vectors SHALL differ, because area carries the size

### Requirement: Appearance is normalized by the cell's own pixels

The system SHALL normalize each cell's appearance by the mean and standard deviation of
that cell's own non-zero pixels, measured before the crop is resampled, and SHALL leave
the background zero. A cell's overall brightness scale SHALL therefore not affect its
appearance. The appearance encoder normalizes each crop again; neither step preserves
absolute brightness.

#### Scenario: A crop is zero-mean and unit-spread over the cell

- **WHEN** a textured cell of exactly the crop size is extracted, so that no resampling occurs
- **THEN** the mean of its non-zero appearance pixels SHALL be zero
- **AND** their standard deviation SHALL be one, to within the stabilizing term

#### Scenario: Overall brightness does not change appearance

- **WHEN** two cells have identical shape and texture but one is three times brighter
- **THEN** their appearance crops SHALL be equal to within the stabilizing term

#### Scenario: A cell of one uniform value has no spread to normalize

- **WHEN** a cell's non-zero pixels all share one value
- **THEN** its appearance SHALL be all zeros, because its standard deviation is zero

### Requirement: Centroids are reported in image coordinates

Centroids SHALL be reported in the coordinate frame of the rescaled image, with no
offset. A consumer comparing a centroid against an image coordinate SHALL NOT need to
correct it.

#### Scenario: A cell at a known image position

- **WHEN** a frame contains a single cell centred at image coordinate `(19.5, 19.5)`
- **THEN** its reported centroid SHALL be `(19.5, 19.5)`

#### Scenario: Padding entries remain at the origin

- **WHEN** a frame contains fewer cells than the padded array allows
- **THEN** the unused entries' centroids SHALL remain `(0, 0)`
- **AND** a real cell near the image origin MAY lie close to that value, so consumers
  SHALL distinguish padding by its all-zero morphology rather than by its position

### Requirement: An all-zero appearance crop is left unnormalized

When a cell's masked appearance contains no non-zero pixels, the system SHALL leave the
crop as zeros. It SHALL NOT emit a warning and SHALL NOT produce NaN. The guard that
decides whether there is anything to normalize SHALL test whether the set of non-zero
pixels is empty. This corrects divergence 14, where the guard tested the arity of an
index tuple, was always true, and emitted `RuntimeWarning: Mean of empty slice`.

#### Scenario: A cell sits on an all-zero image region

- **WHEN** a labelled cell covers a region where the movie is entirely zero
- **THEN** no `RuntimeWarning` SHALL be emitted
- **AND** the resulting appearance crop SHALL be all zeros rather than NaN
