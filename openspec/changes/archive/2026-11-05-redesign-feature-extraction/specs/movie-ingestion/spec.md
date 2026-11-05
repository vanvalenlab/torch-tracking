# Spec Delta

## ADDED Requirements

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

## MODIFIED Requirements

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

## REMOVED Requirements

### Requirement: Appearance is a fixed-size crop that clips large cells

**Reason**: Superseded. Once `correct-scale-convention` corrected the rescaling rule, the
fixed 32×32 window clipped 47.4% of cells, each losing a median 29% of its bounding box.
The window also let absolute cell size into the appearance channel, which belongs to
morphology. Replaced by "Appearance is an aspect-preserving resize to a fixed size". The
requirement carried a **Pending-fix** marker naming this change.

**Migration**: Consumers relying on a crop being a literal window onto the image, at
image scale and centred on the centroid, MUST stop. A crop is now a resampled view of one
cell, normalized in size. The scenario "A small cell is padded, not magnified" is
deliberately inverted: small cells are now magnified, 51.3% of cells being smaller than
the crop. The characterization test pinning the clipping behavior is deleted with this
change.

### Requirement: Each cell yields appearance, morphology and centroid features

**Reason**: Replaced by "Every cell yields appearance, morphology and centroid features",
which states the same obligation without the crop-mode and normalization-flag conditions.
The old requirement specified that masking a crop to its own cell was governed by the
normalization flag in fixed-crop mode but unconditional in resize mode. Both flags are
deleted by this change, so the asymmetry and the scenario pinning it have nothing left to
describe.

**Migration**: Callers passing `crop_mode` or `norm` MUST drop those arguments. Masking
is now unconditional, so a caller that passed `norm=False` to obtain an unmasked crop in
fixed-crop mode will now receive a masked one. The characterization test pinning the
asymmetry is updated to assert unconditional masking.

### Requirement: Centroids are reported in padded-crop coordinates

**Reason**: Moot. The `+32` offset was an artifact of the zero-padding step the fixed
window required in order to crop near an image edge. The aspect-preserving resize works
from each cell's bounding box and needs no padding, so the offset does not arise. This is
divergence 11, which this change makes moot rather than fixes — the defective code path
no longer exists. Replaced by "Centroids are reported in image coordinates".

**Migration**: Any consumer that subtracted 32 from a reported centroid to recover an
image coordinate MUST stop doing so. Consumers that used centroid differences rather than
absolute positions are unaffected. Padding entries still sit at `(0, 0)`, but real cells
may now lie near that value — 5.1% of cells in the test split sit within 10 px of it — so
padding MUST be identified by its all-zero morphology. The characterization test pinning
the offset is deleted with this change.
