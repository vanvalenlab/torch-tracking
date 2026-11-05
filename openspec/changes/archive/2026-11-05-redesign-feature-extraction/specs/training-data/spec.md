# Spec Delta

## ADDED Requirements

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

## MODIFIED Requirements

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

## REMOVED Requirements

### Requirement: Crop mode is configurable in name only

**Reason**: Removed by deletion rather than by repair. The requirement existed because
the training configuration recorded a `crop_mode` value that nothing consumed, which
misdescribed the stored data. After this change there is exactly one appearance
extraction behavior, so a parameter selecting between behaviors has nothing to select.
`crop_mode` is deleted from the training configuration, from the dataloader's recorded
configuration and from the feature-extraction interface. This closes divergence 18.

**Migration**: Any configuration file or script setting `crop_mode` MUST drop the key;
it is no longer read or recorded. Crop size remains consumed when the model is built, and
a crop size disagreeing with the stored appearances remains a silent misconfiguration, so
that hazard is unchanged by this removal. The `norm` parameter is removed on the same
grounds: masking and normalization are now both unconditional, so it has nothing left to
switch.

### Requirement: Appearance is a fixed-size crop that clips large cells

**Reason**: Superseded. The fixed 32×32 window clipped 47.4% of cells once
`correct-scale-convention` corrected the rescaling rule, discarding a median 29% of each
clipped cell's bounding box. It is replaced by the aspect-preserving resize specified
above. The requirement carried a **Superseded-pending** marker naming this change, and no
characterization test was written against it on the conversion side.

**Migration**: Every existing processed zarr MUST be regenerated. Array names and shapes
are unchanged — `appearances` remains `(B, T, N, 32, 32, 1)` and `centroids` remains
`(B, T, N, 2)` — so a stale zarr is not detected automatically; only the stored values
change. Training against a zarr produced before this change will silently train on
clipped crops with offset centroids. Any checkpoint trained on such data is
invalidated.
