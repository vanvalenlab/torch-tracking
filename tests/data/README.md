# Test fixtures

Nothing in this directory derives from any dataset or from any trained model.
That is the point: it lets the end-to-end (L2) tests run from a clean checkout with no
credentials, no dataset, and no gated weights.

## `random_init_seed0.pt` — this is NOT a model

**Do not use this file for inference. It will produce meaningless output.**

It contains untrained, randomly initialized weights: the result of constructing
`GNNTrackingModel` under `torch.manual_seed(0)` and saving it immediately, before any
training. It has learned nothing. It carries no information from
`DynamicNuclearNet-tracking` or from any other data, and it has no scientific value.

| | |
| --- | --- |
| Size | 1.17 MB |
| SHA256 | `a5c7e4088663cd7f8a8f68a2969638d8e2f0b400e17ca91c9d09ccadb4041f34` |
| Seed | 0 |
| Architecture | `gat`, `encoder_dim=64`, `n_layers=2`, `crop_size=32` — matching what `CellTracker` hardcodes |
| Regenerate | `python tests/data/make_random_init.py` |

### Why it is committed, and why that is consistent with constraint 6

`VISION.md` constraint 6 states that **trained weights never leave the HuggingFace
`vanvalenlab` org** — never committed, never attached to a release, never otherwise
redistributed. That constraint governs trained weights. This file is not one: it is noise
with the shape of a checkpoint, and it conveys nothing that the license on the real
weights exists to protect.

It is committed rather than regenerated at test time because `torch.manual_seed`
reproducibility is not guaranteed across torch versions or platforms. Regenerating it on
the fly would mean a torch upgrade could break the L2 goldens for reasons unrelated to
any code change — a confusing failure with no real cause.

The real weights remain gated and referenced by revision and hash only; see
`metrics/PROVENANCE.md` and `doc/model_access.md`.

## `synthetic_movie.py` — a procedurally generated time-lapse

Four elliptical nuclei drifting across a 160×160 field over 10 frames, one of which
divides at frame 5. No RNG is involved, so it is byte-identical on every run and every
platform.

It exercises the full tracking decision path — ingestion, embedding, the cost matrix,
the linear assignment, division detection, and lineage construction — without touching
a dataset.

**It measures nothing about accuracy.** Tracking this movie with untrained weights says
nothing about whether the model is any good; that is the L3 layer's job, and L3 needs
both gated assets. L2 is a change-detector, not a benchmark. A green L2 run is not
evidence that the model still works.

Its ground-truth lineage is keyed by **str** label, matching the `.trk` format and what
`is_valid_lineage` expects. `CellTracker.get_lineage_dict()` returns **int** keys
instead; that mismatch is recorded as divergence 1 in the campaign's divergence register
and belongs to the `lineage-output` capability.
