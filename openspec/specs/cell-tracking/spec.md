# cell-tracking Specification

## Purpose

Turns the per-cell features and embeddings produced by `movie-ingestion` into tracks: it
decides, frame by frame, which cell continues which track, which tracks end, which cells
begin new ones, and which cells are daughters of a dividing parent. It owns the resulting
tracks and every shape they are exported in, and it stops short of scoring them.

## Requirements

### Requirement: Tracking begins with every cell in the first frame

The system SHALL create one track for each cell present in the first frame before any
assignment is performed. No cell in the first frame may be left untracked, and no track
may exist that is not seeded from a cell.

#### Scenario: The first frame holds several cells

- **WHEN** tracking starts on a movie whose first frame holds four labelled cells
- **THEN** four tracks SHALL exist before the second frame is processed
- **AND** each SHALL record frame `0` as its only frame so far

#### Scenario: The first frame holds no cells

- **WHEN** tracking starts on a movie whose first frame holds no labelled cells
- **THEN** no tracks SHALL be created from it
- **AND** tracking SHALL proceed to the next frame without error

### Requirement: Track numbers are assigned by one rule and are gapless

The system SHALL number tracks consecutively from `0`, assigning each new track the number
one greater than the highest in use, and SHALL never reuse or skip a number. A track's
exported `label` SHALL be its number plus one, so labels run from `1`. This corrects
divergence 8, where new numbers were derived as `len(tracks)` in two places and
`max(tracks)` in a third; those agree only while numbering is gapless, which was never
stated.

#### Scenario: A track is created when others already exist

- **WHEN** three tracks numbered `0`, `1` and `2` exist and a new cell appears
- **THEN** the new track SHALL be numbered `3`
- **AND** its exported label SHALL be `4`

#### Scenario: Numbering is derived identically wherever tracks are created

- **WHEN** a track is created as a birth and a track is created by division handling
- **THEN** both SHALL obtain their number from the same rule
- **AND** the set of track numbers SHALL remain the integers `0` through `n-1`

### Requirement: A track always holds at least one frame

The system SHALL maintain the invariant that every track known to it records at least one
frame. Removing a track's last remaining frame SHALL raise rather than leave an empty
track in place. The invariant exists because track history, cost-matrix rows and lineage
export are all derived from a track's frames, and an empty track is indistinguishable from
a live one by number alone.

#### Scenario: Division handling would empty a track

- **WHEN** division handling removes the only frame a track records
- **THEN** the system SHALL raise an error identifying that track
- **AND** it SHALL NOT silently retain a track with no frames

#### Scenario: Division handling leaves a frame behind

- **WHEN** division handling removes one of a track's two recorded frames
- **THEN** that track SHALL survive with its remaining frame
- **AND** no error SHALL be raised

### Requirement: Track history is addressed by track number, never by position

The system SHALL key a track's fetched history by that track's number. It SHALL NOT key
history by the position a track happens to occupy in an iteration, because consumers use
those keys to look tracks up. This corrects divergence 35: history was keyed by
enumeration position while the cost matrix and division detection read those keys as track
numbers, so a track excluded from the fetch shifted every later track's identity by one
without any error.

#### Scenario: Some tracks are excluded from a fetch

- **WHEN** history is fetched and one track is excluded from the result
- **THEN** the remaining entries SHALL still be keyed by their own track numbers
- **AND** every key SHALL resolve to the track whose history it holds

#### Scenario: Keys are used to resolve tracks

- **WHEN** the cost matrix or division detection resolves a track from a history key
- **THEN** the track it resolves SHALL be the one whose history that entry contains

### Requirement: Track history is the configured number of preceding frames

The system SHALL supply the model with exactly `track_length` frames of a track's history,
taken from the frames strictly preceding the frame being tracked and ending with the most
recent. Where a track has too few, the history SHALL be padded to length by repeating its
most recent frame.

#### Scenario: A track is older than the history window

- **WHEN** a track has twelve preceding frames and `track_length` is `8`
- **THEN** the history SHALL hold the eight most recent of them, in order

#### Scenario: A track is younger than the history window

- **WHEN** a track has three preceding frames and `track_length` is `8`
- **THEN** the history SHALL hold eight entries
- **AND** the final five SHALL repeat the most recent of the three

#### Scenario: A track was born in the frame being tracked

- **WHEN** a track's only frame is the frame currently being tracked
- **THEN** it SHALL contribute no history for that frame, having no preceding frames

### Requirement: The history length defaults to the window the model was trained on

`track_length` SHALL default to `8`, the window the processed training data is chunked
for. The two SHALL be documented as coupled: changing one without the other presents the
model a history unlike anything it saw in training. This corrects divergence 7, where the
documentation described a seven-frame history.

#### Scenario: The default is left unset

- **WHEN** a caller does not configure `track_length`
- **THEN** eight frames of history SHALL be supplied per track

#### Scenario: Documentation is read as a description of the history

- **WHEN** the documented history length is compared with the implemented default
- **THEN** they SHALL agree

### Requirement: The model scores each track against each cell in three classes

The model SHALL return, for every pairing of a tracked history with a cell in the current
frame, a probability over three classes in a fixed order: index `0` that they are
different cells, index `1` that they are the same cell, and index `2` that the cell is a
daughter of the track. This ordering is fixed by the training targets and SHALL be
documented as such. This corrects divergence 36, where the documentation named the first
two classes in the opposite order.

#### Scenario: Class ordering is relied on by the cost matrix

- **WHEN** linking costs are derived from the model's output
- **THEN** they SHALL be derived from index `1`, the probability of being the same cell

#### Scenario: Class ordering is relied on by division detection

- **WHEN** a parent is sought for a newly appeared cell
- **THEN** the probability consulted SHALL be index `2`, the daughter probability

### Requirement: Linking cost falls as the probability of being the same cell rises

The system SHALL assign each track-to-cell pairing a cost of one minus the probability
that they are the same cell, so that the cheapest assignment is the most likely
continuation.

#### Scenario: A confident continuation

- **WHEN** the model reports a `0.98` probability that a cell continues a track
- **THEN** that pairing's cost SHALL be `0.02`

#### Scenario: A track that has already divided

- **WHEN** a track has been capped because it divided
- **THEN** every pairing in its row SHALL be set to the maximum cost of `1.0`
- **AND** continuing it SHALL therefore be no cheaper than ending it

### Requirement: Assignment prices continuation, birth and death together

The system SHALL solve one assignment problem per frame over a square cost matrix that
offers every track a choice between continuing into a cell and ending, and offers every
cell a choice between continuing a track and beginning a new one. Ending a track SHALL
cost `death` and beginning a track SHALL cost `birth`; both SHALL be strictly less than
the maximum linking cost of `1.0`, so that a capped track is always cheaper to end than to
continue.

#### Scenario: A cell has no plausible predecessor

- **WHEN** every track's linking cost for a cell exceeds `birth`
- **THEN** that cell SHALL begin a new track rather than continue an existing one

#### Scenario: A track has no plausible successor

- **WHEN** every cell's linking cost for a track exceeds `death`
- **THEN** that track SHALL end rather than continue

#### Scenario: Thresholds are configured at or above the maximum cost

- **WHEN** `birth` or `death` is configured at `1.0` or higher
- **THEN** the system SHALL reject the configuration, naming the threshold and the bound

### Requirement: Threshold defaults are documented as implemented

The system's documented default thresholds SHALL match its implemented ones. The
implemented defaults are `death` `0.999`, `birth` `0.999` and `division` `0.5`. This
corrects divergence 2, where the documentation advertised `birth` `0.99` and `division`
`0.05`, the latter differing from the implementation by a factor of ten.

#### Scenario: Documentation is read as configuration guidance

- **WHEN** a caller configures thresholds from the documented defaults
- **THEN** the resulting behavior SHALL match leaving them unset

### Requirement: A newly appeared cell may be attributed to a dividing parent

When a cell begins a new track, the system SHALL look for a parent among the tracks scored
against that cell, choosing the track with the highest daughter probability above
`division`. A track SHALL NOT be eligible as a parent if it has already been capped, or if
the frame being tracked is the only frame it records.

#### Scenario: A confident division

- **WHEN** a newly appeared cell is scored with daughter probability `0.9` against one
  uncapped track and `division` is `0.5`
- **THEN** that track SHALL be recorded as the cell's parent
- **AND** the cell's track SHALL be recorded among that parent's daughters

#### Scenario: No track clears the threshold

- **WHEN** no track's daughter probability for a newly appeared cell exceeds `division`
- **THEN** the cell's track SHALL record no parent

#### Scenario: A track born in this frame is not a parent

- **WHEN** the only frame a track records is the frame being tracked
- **THEN** it SHALL NOT be selected as a parent, whatever its daughter probability

### Requirement: A parent may have more than two daughters

The system SHALL NOT limit the number of daughters a track may have. Multipolar mitosis
occurs in the data this package is trained and evaluated on, so a parent with three or
more daughters is a valid observation rather than an error to be corrected. Any
daughter-count limit belongs to a data-quality filter, not to tracking.

#### Scenario: A third daughter is attributed to a parent

- **WHEN** a third cell is attributed to a track that already records two daughters
- **THEN** it SHALL be added to that track's daughters
- **AND** no error or warning SHALL be raised on the ground of daughter count

#### Scenario: Lineage export carries every daughter

- **WHEN** a track records three daughters
- **THEN** its exported lineage entry SHALL list all three

### Requirement: A divided track ends at the division

The system SHALL cap a track once it has daughters, recording the frame in which the
division was detected, and SHALL NOT extend a capped track into later frames. A capped
track that the assignment nonetheless paired with a cell SHALL surrender that frame, and
the cell SHALL become a further daughter of it.

#### Scenario: A capped track is paired with a cell

- **WHEN** assignment pairs a capped track with a cell in the current frame
- **THEN** that frame SHALL be removed from the capped track
- **AND** the cell SHALL begin a new track recorded as a daughter of it

#### Scenario: A capped track records its division frame

- **WHEN** a track is capped in frame `5`
- **THEN** its recorded division frame SHALL be `5`

### Requirement: Tracked masks are relabelled once, from the original resolution

The system SHALL produce tracked masks by applying the accumulated cell-to-track
relabelling to the masks at the resolution the caller supplied, not to the rescaled copy
used for feature extraction. Relabelling SHALL be applied as a single pass after tracking
completes. A label with no recorded assignment in a frame SHALL be left unchanged.

#### Scenario: The movie was rescaled for tracking

- **WHEN** a movie was rescaled during ingestion and then tracked
- **THEN** the tracked masks SHALL have the caller's original height and width

#### Scenario: A label was never assigned

- **WHEN** a frame holds a label that tracking never assigned to a track
- **THEN** that label SHALL appear unchanged in the tracked masks

### Requirement: Lineage keys are integers

The system SHALL key the exported lineage dictionary by integer cell label, and each
entry's `label` SHALL equal its key. Integers are the in-memory convention throughout this
package; conversion to and from the strings a JSON document requires SHALL happen only at
that boundary. This corrects divergence 1, where the exported lineage was keyed by integer
while lineage validation resolved parents and daughters as strings, so validation reported
every cell of a tracker-produced lineage as invalid.

#### Scenario: A lineage is exported and validated

- **WHEN** an exported lineage is passed to lineage validation together with its masks
- **THEN** validation SHALL resolve every parent and daughter reference
- **AND** a lineage consistent with its masks SHALL be reported valid with no warnings

#### Scenario: A consumer receives an exported lineage

- **WHEN** a consumer reads the exported lineage
- **THEN** its keys SHALL be integers
- **AND** no consumer SHALL need to re-key it before use

### Requirement: A lineage entry records the track's identity, extent and relationships

Each lineage entry SHALL record the track's label, the frames in which it appears, its
parent's label or nothing, the labels of its daughters, the frame in which it divided or
nothing, and whether it is capped. Parent and daughter references SHALL be labels, in the
same space as the keys, never internal track numbers.

#### Scenario: A track that never divided

- **WHEN** a track appears in frames `0` through `4` and never divides
- **THEN** its entry SHALL list those frames, no daughters, and no division frame

#### Scenario: A daughter names its parent

- **WHEN** a track's parent is the track exported with label `3`
- **THEN** that track's entry SHALL record parent `3`
- **AND** label `3`'s entry SHALL list this track among its daughters

#### Scenario: A track with no parent

- **WHEN** a track began as a birth rather than a division
- **THEN** its entry SHALL record no parent rather than a sentinel number

### Requirement: Tracking the same input twice gives the same result

Given the same movie, masks, checkpoint and parameters, the system SHALL produce identical
tracks, identical tracked masks and an identical lineage. No part of the tracking decision
path may depend on wall-clock time, iteration order that varies between runs, or
uninitialized memory.

#### Scenario: A movie is tracked twice in one process

- **WHEN** the same movie and masks are tracked twice with the same checkpoint and
  parameters
- **THEN** both runs SHALL produce equal lineages
- **AND** both runs SHALL produce identical tracked masks
