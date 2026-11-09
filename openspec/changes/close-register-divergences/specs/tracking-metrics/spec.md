# Spec Delta

## ADDED Requirements

### Requirement: Lineage-to-graph conversion has one definition

Conversion of a lineage into its graph representation SHALL exist once in the package, in
the module that evaluates against it. No other module SHALL carry a copy, whether or not
anything calls it. A second definition invites the two to drift and misleads a reader about
which one the recorded figures came from. This closes divergence 6.

#### Scenario: Only one module defines the conversion

- **WHEN** the package is searched for a definition converting a lineage into a graph
- **THEN** exactly one SHALL be found
- **AND** it SHALL live in the metrics module

#### Scenario: Removing the copy changes no reachable behavior

- **WHEN** the evaluation harness converts a lineage
- **THEN** it SHALL produce the same graph it produced before the duplicate was removed

#### Scenario: Imports serving only the removed copy go with it

- **WHEN** a module no longer defines the conversion
- **THEN** any import that served only that definition SHALL be removed from it

### Requirement: The evaluation command line defaults to what its function defaults to

Every tracking parameter the evaluation command line exposes SHALL default to the value the
evaluation function and the tracker use when the caller supplies nothing. A command line
that silently substitutes a different value makes an unflagged invocation measure a
configuration nobody chose, and the figures it produces cannot be told from the recorded
ones by inspection. This closes divergence 77.

#### Scenario: An unflagged run matches a defaulted call

- **WHEN** evaluation is invoked from the command line with no tracking parameters given
- **THEN** every tracking parameter SHALL take the value the evaluation function would use
  if called without it

#### Scenario: The match distance default is the retuned one

- **WHEN** the command line is invoked without a distance threshold
- **THEN** the threshold SHALL be the package's declared distance threshold, not the value
  inherited from before the resolution correction

#### Scenario: The recorded figures are unaffected

- **WHEN** the reproduction command in the provenance record is run
- **THEN** it SHALL reproduce the committed per-movie results, because it states the
  threshold explicitly

### Requirement: Accuracy tests refuse weights the current code did not produce

The accuracy tests SHALL evaluate only a checkpoint produced by the model code as it now
stands. Where the published checkpoint is known to predate corrections to the model's
arithmetic, the tests SHALL decline to run against it and SHALL say why, rather than
measuring it and reporting the result against the external reference. A caller MAY supply a
local corrected checkpoint instead.

#### Scenario: A superseded published checkpoint is declined

- **WHEN** the accuracy tests run with no checkpoint supplied, and the published checkpoint
  predates corrections to the model's arithmetic
- **THEN** the tests SHALL skip without downloading it
- **AND** the skip message SHALL name the corrections it predates and how to supply a
  corrected checkpoint

#### Scenario: A supplied checkpoint is used as given

- **WHEN** a local checkpoint is supplied to the accuracy tests
- **THEN** that checkpoint SHALL be evaluated
- **AND** nothing SHALL be downloaded

#### Scenario: A credential-free run stays green

- **WHEN** the accuracy tests run without credentials, a dataset, or a supplied checkpoint
- **THEN** they SHALL skip rather than fail

#### Scenario: Republished weights restore the download

- **WHEN** a checkpoint produced by the current model code has been published
- **THEN** the tests SHALL identify it by repository, revision and hash
- **AND** the provenance record SHALL name the same revision and hash
