# Spec Delta

## MODIFIED Requirements

### Requirement: Appearance is a fixed-size crop that clips large cells

**Pending-fix:** `redesign-feature-extraction` replaces this with an aspect-preserving
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

**Pending-fix:** `redesign-feature-extraction` makes this moot. Centroids SHALL be reported
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

**Pending-fix:** `redesign-feature-extraction` makes this moot by replacing the fixed-crop
branch it lives in. When a cell's appearance crop contains no non-zero pixels, the system
SHALL emit `RuntimeWarning: Mean of empty slice` and SHALL leave the crop as zeros rather
than normalizing it. The guard intended to prevent this tests the arity of an index tuple
rather than whether it is empty, so it never prevents anything. This is divergence 14.

#### Scenario: A cell sits on an all-zero image region

- **WHEN** a labelled cell covers a region where the movie is entirely zero
- **THEN** a `RuntimeWarning` SHALL be emitted
- **AND** the resulting appearance crop SHALL be all zeros rather than NaN
