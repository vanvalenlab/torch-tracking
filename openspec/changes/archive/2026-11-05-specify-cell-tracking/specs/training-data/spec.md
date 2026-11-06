# Spec Delta

## MODIFIED Requirements

### Requirement: Labels are relabelled sequentially per movie

The system SHALL relabel each movie's masks so labels are sequential from 1, and SHALL
rewrite that movie's lineage dictionary to the new labels. Keys of the rewritten lineage
SHALL be integers, and each entry's `label` SHALL equal its key. The input lineage SHALL
be accepted keyed by integers or by the strings a JSON document yields, so a lineage this
package produced can be relabelled again. This closes the input half of divergence 1.

#### Scenario: Non-sequential labels are compacted

- **WHEN** a movie's masks use labels `[1, 5, 9]`
- **THEN** the relabelled masks SHALL use `[1, 2, 3]` and the lineage SHALL be keyed `1, 2, 3`

#### Scenario: Parent references follow the relabelling

- **WHEN** a cell's `parent` is a label that was remapped
- **THEN** the rewritten entry's `parent` SHALL be the remapped value

#### Scenario: A null parent is preserved

- **WHEN** a cell's `parent` is `null`
- **THEN** the rewritten entry's `parent` SHALL remain `null` rather than becoming an integer

#### Scenario: A lineage read from JSON has string keys

- **WHEN** a lineage decoded from a `.trk` archive is keyed by strings
- **THEN** relabelling SHALL resolve every entry
- **AND** the rewritten lineage SHALL be keyed by integers

#### Scenario: A lineage already keyed by integers is relabelled

- **WHEN** a lineage keyed by integers is passed to relabelling
- **THEN** it SHALL be resolved without error
- **AND** the rewritten lineage SHALL be keyed by integers
