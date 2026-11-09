# Spec Delta

## MODIFIED Requirements

### Requirement: Padding slots are isolated inside the graph

A padding slot SHALL NOT exchange information with a real cell through the graph. The
adjacency the graph layers consume SHALL contain no edge joining a padding slot to a real
cell, SHALL give a padding slot no edge at all, and SHALL carry no weighting that is
computed and then discarded.

#### Scenario: No real-to-padding edge exists

- **WHEN** a batch containing padding slots is encoded
- **THEN** the edge set passed to the graph layers contains no edge joining a padding slot to a real cell

#### Scenario: Edge weights are either used or not computed

- **WHEN** the adjacency is prepared for the graph layers
- **THEN** any weighting applied to it is reflected in what the graph layers receive
- **AND** a weighting that is computed and then discarded does not remain in the code

#### Scenario: A padding slot carries no self-loop

- **WHEN** the edge set passed to the graph layers is inspected
- **THEN** no padding slot has an edge to itself
- **AND** a real cell's self-loop is unaffected

#### Scenario: A padding slot has no edges at all

- **WHEN** a batch containing padding slots is encoded
- **THEN** every edge in the set passed to the graph layers has a real cell at both ends

#### Scenario: An isolated slot does not poison the graph output

- **WHEN** a batch is encoded in which some slots have no edges
- **THEN** every real cell's graph output is finite
- **AND** no real cell's graph output depends on how many isolated slots the batch carries

### Requirement: Appearance crops are normalized as whole images

With the `whole_image` method, each appearance crop SHALL be normalized over its full
spatial extent, so the crop as a whole has zero mean and unit variance. Sub-regions of a
crop SHALL NOT be normalized independently. The spatial extent SHALL be determined by the
input layout the caller declares, not by an assumed layout.

#### Scenario: A normalized crop has whole-image statistics

- **WHEN** a crop is passed through appearance normalization with the `whole_image` method
- **THEN** the returned crop has approximately zero mean and unit variance over its full spatial extent

#### Scenario: Sub-regions are not normalized independently

- **WHEN** a crop whose rows have different means is normalized
- **THEN** the returned rows do not all have zero mean
- **AND** the relative intensity differences between rows are preserved

#### Scenario: The declared layout is the layout used

- **WHEN** the same crop is normalized once under each supported input layout, each declared correctly
- **THEN** the two results agree, up to the transposition between the layouts

#### Scenario: An unsupported layout is refused

- **WHEN** normalization is constructed with a layout name it does not support
- **THEN** construction raises, naming the layouts it accepts
- **AND** it does not silently fall back to one of them

### Requirement: Normalization statistics exclude padding slots

Normalization layers SHALL compute their statistics over real cells only. A padding slot
SHALL NOT contribute to any running mean or variance, nor shift the statistics a real cell
is normalized by. Where a layer normalizes pairs of slots, a pair SHALL count as real only
when both of its ends are real.

#### Scenario: Padding does not move a real cell's normalized value

- **WHEN** the same real cells are encoded in two batches that differ only in how many padding slots they carry
- **THEN** each real cell receives the same normalized features in both

#### Scenario: Padding is identified by its content, not its position

- **WHEN** a padding slot is distinguished from a real cell
- **THEN** it is identified by its all-zero morphology
- **AND** it is not identified by its centroid, which augmentation translates during training

#### Scenario: Padding does not move the model's graded output

- **WHEN** the same real cells are scored in two batches that differ only in how many padding slots they carry
- **THEN** every entry the loss grades receives the same value in both

#### Scenario: Running statistics are accumulated over real cells only

- **WHEN** batches containing padding slots are passed through in training mode
- **THEN** each normalization layer's accumulated running statistics equal those it would accumulate from the same real cells with no padding present

#### Scenario: A batch of only padding does not fault

- **WHEN** a batch is encoded in which every slot is padding
- **THEN** the forward pass completes without raising
- **AND** no running statistic is updated from it

#### Scenario: Correcting the statistics does not change the checkpoint's shape

- **WHEN** a checkpoint saved before this correction is loaded into the corrected model
- **THEN** every parameter and buffer name and shape matches
- **AND** loading succeeds without a shape error

## ADDED Requirements

### Requirement: Temporal merging treats each track as its own sequence

The temporal merge SHALL run its recurrent layer over each track's frames in time order,
as one sequence per batch entry and track slot. A track's merged output SHALL depend only
on that track's own frames, never on any other slot in the batch.

#### Scenario: Each track is its own sequence

- **WHEN** a batch of several tracks over several frames is temporally merged
- **THEN** each track's output equals what the recurrent layer gives that track's frames run alone, in time order

#### Scenario: Other slots do not move a track's output

- **WHEN** the same real tracks are merged in two batches that differ only in how many padding slots they carry
- **THEN** each real track receives the same merged output in both

#### Scenario: A single track is unaffected

- **WHEN** a batch holds exactly one track
- **THEN** the merged output is the recurrent layer applied to that track's frames in time order
