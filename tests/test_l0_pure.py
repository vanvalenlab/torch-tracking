"""L0 - pure functions. Needs nothing: no checkpoint, no dataset, no GPU, no network.

This layer covers the functions that transform arrays and lineages without a forward
pass, which is a large share of the codebase:

    clean_up_annotations, relabel_sequential_lineage, is_valid_lineage, resize,
    histogram_normalization, get_image_features, get_temporal_adjacency,
    compute_deltas, CellTracker._build_cost_matrix, CellTracker._apply_relabel_map,
    and effectively all of metrics.py

Assertions are deliberately absent for now. They arrive with the per-capability
`characterize-*` changes, next to the requirements they verify, so that a failing test
always points at a written requirement. See openspec/changes/ and tests/README.md.
"""

import pytest

pytestmark = pytest.mark.l0
