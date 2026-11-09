# Spec Delta

## Purpose

The graph neural network that scores whether a tracked history and a cell in the next frame
are the same cell, different cells, or a mother/daughter pair. This capability owns the
architecture, how padding slots travel through it, which input layout it accepts, and how a
trained checkpoint is located, loaded and placed on a device. It does not own the
assignment algorithm that consumes its probabilities; that is `cell-tracking`.

## ADDED Requirements

### Requirement: Channels-last is the only supported input layout

The model SHALL accept appearance crops in channels-last layout and SHALL NOT offer a
channels-first option it cannot honor. The training branch unpacks a channels-last
six-dimensional appearance tensor, so channels-first cannot reach a forward pass. Any
default the constructor exposes SHALL be one the constructor can actually build.

#### Scenario: The default configuration constructs

- **WHEN** the model is constructed with no `data_format` argument
- **THEN** construction succeeds and the model accepts channels-last appearances

#### Scenario: An unsupported layout is refused at construction

- **WHEN** the model is constructed with a `data_format` other than `channels_last`
- **THEN** construction raises `ValueError` naming the supported layout
- **AND** the failure occurs at construction, not at the first forward pass

### Requirement: Configuration is validated before it is used

The model SHALL validate every configuration value it branches on, and SHALL perform
structural checks before indexing the structure they check. Error messages SHALL state the
requirement that was violated and the value received.

#### Scenario: Shape arity is checked before the shape is indexed

- **WHEN** `appearance_shape` has a length other than three
- **THEN** construction raises `ValueError` reporting the required length of three
- **AND** the message does not claim a different required length

#### Scenario: No configuration path leaves a comparison unbound

- **WHEN** any accepted configuration is supplied
- **THEN** every value the validator compares has been assigned
- **AND** no configuration reaches `UnboundLocalError`

#### Scenario: Spatial dimensions must be square and a power of two

- **WHEN** `crop_size` is not a power of two
- **THEN** construction raises `ValueError` naming the power-of-two requirement

### Requirement: A normalization choice is either honored or not offered

Every module that accepts a `norm_layer` argument SHALL build the normalization that
argument names, in every layer it owns. A module that cannot honor the choice SHALL NOT
accept the argument. Accepting a value and silently building something else SHALL NOT
occur.

#### Scenario: Every module honors the normalization it is given

- **WHEN** the model is constructed with a given `norm_layer`
- **THEN** no module contains a normalization layer of a different kind than requested

#### Scenario: A requested normalization produces a working forward pass

- **WHEN** the model is constructed with a `norm_layer` the constructor accepts
- **THEN** a training forward pass and an inference forward pass both complete
- **AND** neither raises from a normalization layer applied to the wrong tensor rank

### Requirement: Feature widths are derived from the modules that produce them

Every layer that consumes concatenated features SHALL size its input by summing the actual
output widths of the modules feeding it. A layer SHALL NOT assume those widths are equal or
derive them from a single configuration value that happens to coincide with them.

#### Scenario: Encoder widths that differ still build a working model

- **WHEN** the model is constructed so that the encoder width and the embedding width differ
- **THEN** construction succeeds
- **AND** a training forward pass completes without a shape mismatch

#### Scenario: Widths are summed, not multiplied by a count

- **WHEN** a layer consumes the output of several modules
- **THEN** its input width equals the sum of those modules' reported output widths

### Requirement: Padding slots are isolated inside the graph

A padding slot SHALL NOT exchange information with a real cell through the graph. The
adjacency the graph layers consume SHALL contain no edge joining a padding slot to a real
cell, and SHALL carry no weighting that is computed and then discarded.

**Pending fix.** A padding slot still receives a self-loop. Removing it changes padding
embeddings, and so the statistics divergence 34 concerns; `correct-model-normalization`
completes it. Divergence 60.

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

### Requirement: Appearance crops are normalized as whole images

With the `whole_image` method, each appearance crop SHALL be normalized over its full
spatial extent, so the crop as a whole has zero mean and unit variance. Sub-regions of a
crop SHALL NOT be normalized independently.

**Pending fix.** The implementation normalizes each crop row by row. Correcting it changes
every appearance feature, so it needs a retrain; `correct-model-normalization` completes
it. Divergence 80.

#### Scenario: A normalized crop has whole-image statistics

- **WHEN** a crop is passed through appearance normalization with the `whole_image` method
- **THEN** the returned crop has approximately zero mean and unit variance over its full spatial extent

#### Scenario: Sub-regions are not normalized independently

- **WHEN** a crop whose rows have different means is normalized
- **THEN** the returned rows do not all have zero mean
- **AND** the relative intensity differences between rows are preserved

### Requirement: Normalization statistics exclude padding slots

Normalization layers SHALL compute their statistics over real cells only. A padding slot
SHALL NOT contribute to any running mean or variance, nor shift the statistics a real cell
is normalized by.

**Pending fix.** Fifteen BatchNorm layers take statistics over padding. Training-path only;
inference runs in eval mode with no padding. The fix needs statistics re-estimated and a
retrain; `correct-model-normalization` completes it. Divergence 34.

#### Scenario: Padding does not move a real cell's normalized value

- **WHEN** the same real cells are encoded in two batches that differ only in how many padding slots they carry
- **THEN** each real cell receives the same normalized features in both

#### Scenario: Padding is identified by its content, not its position

- **WHEN** a padding slot is distinguished from a real cell
- **THEN** it is identified by its all-zero morphology
- **AND** it is not identified by its centroid, which augmentation translates during training

### Requirement: A checkpoint is located, loaded and placed explicitly

The model's weights SHALL be loaded from a path the caller supplies, and every parameter
naming that path SHALL describe what it actually accepts. The device the model runs on
SHALL be the one documented, and CPU SHALL remain the default so that inference needs no
GPU.

#### Scenario: The checkpoint parameter names what it accepts

- **WHEN** a caller reads the documentation for the checkpoint parameter
- **THEN** it describes a file path to a checkpoint
- **AND** the documented meaning matches what the code passes to the loader

#### Scenario: CPU is the documented and implemented default

- **WHEN** a tracker is constructed with no device argument
- **THEN** the model is placed on the CPU
- **AND** the documented default is CPU

#### Scenario: Feature extraction uses the configured crop size

- **WHEN** the crop size is changed from its default
- **THEN** the extracted appearance crops and the arrays sized to hold them both follow the new value
- **AND** no call relies on a separate default that could disagree

### Requirement: The architecture the tracker builds matches the published checkpoint

The tracker SHALL construct the model with the exact architecture the published checkpoint
was trained with, and that coupling SHALL be stated where a reader can find it. A
checkpoint trained with different architecture hyperparameters cannot be loaded into it.

#### Scenario: The hardcoded architecture is recorded

- **WHEN** the tracker constructs its model
- **THEN** the architecture values it uses are recorded in a single shared place that tests read
- **AND** a change to either the tracker or that record fails a test

#### Scenario: Loading a mismatched checkpoint fails loudly

- **WHEN** a checkpoint trained with a different architecture is loaded
- **THEN** loading raises rather than silently producing a differently-shaped model

### Requirement: The documented interface matches the implemented one

Every parameter the documentation describes SHALL exist, every parameter that exists SHALL
be documented, and every documented example SHALL run. Methods SHALL be attributed to the
object that defines them.

#### Scenario: A documented example runs

- **WHEN** a usage example in a docstring or in `docs.md` is executed
- **THEN** it completes without raising
- **AND** it does not pass an argument the target does not accept

#### Scenario: Methods are attributed to their owner

- **WHEN** the documentation describes the forward passes and embedding extraction
- **THEN** it attributes them to the model, which defines them, not to the tracker

#### Scenario: Module self-checks are executable

- **WHEN** a module's runnable self-check block is executed
- **THEN** it completes without raising
- **AND** it constructs each module with arguments that module accepts

### Requirement: The model holds no state it does not use

The model and its modules SHALL NOT build layers, import modules, or assign attributes that
nothing reads. Dead state misleads a reader about what the architecture contains and about
which layers a checkpoint must supply.

#### Scenario: Every built layer is reachable from a forward pass

- **WHEN** the model's modules are enumerated
- **THEN** every layer it builds is used by at least one forward path

#### Scenario: No attribute is assigned and never read

- **WHEN** an attribute is assigned on a model or helper object
- **THEN** at least one code path reads it
