# Tests

The suite is organized into four layers by **what each one needs in order to run**,
rather than by what it tests. With trained weights gated and the dataset currently
unobtainable, that is the distinction that decides where a test can execute.

```text
            +------------------------------------+
  L3  slow  |  accuracy guardrails               |  needs: HF_TOKEN
 maintainer |  real weights + real dataset       |       + the dataset
            +====================================+  <-- the gated boundary
  L2  secs  |  end-to-end tracking               |  needs: nothing
       CI   |  synthetic movie + untrained .pt   |       (fixtures committed)
            +------------------------------------+
  L1   ms   |  model structural contracts        |  needs: torch only
       CI   |  random weights, no checkpoint     |
            +------------------------------------+
  L0   ms   |  pure functions, no forward pass   |  needs: nothing
       CI   |  hand-built arrays                 |
            +------------------------------------+
```

Everything below the gated boundary runs from a clean checkout with no credentials.
That is deliberate: it is what lets a new contributor run the suite, and what makes CI
meaningful on pull requests from forks, where repository secrets are unavailable.

| Layer | Marker | Needs | Runs in CI |
| --- | --- | --- | --- |
| L0 — pure functions | `l0` | Nothing | On demand |
| L1 — model structure | `l1` | `torch` | On demand |
| L2 — end to end | `l2` | Nothing beyond the repository | On demand |
| L3 — accuracy | `l3` | `HF_TOKEN` **and** `DynamicNuclearNet-tracking` | **No** |

The workflow in `.github/workflows/tests.yml` runs L0, L1 and L2. Its automatic triggers
are disabled for now — it runs only on demand, from the Actions tab or
`gh workflow run tests.yml`, until dataset distribution finishes moving to HuggingFace.
Re-enabling it is a two-line edit documented at the top of that file.

## Running them

```bash
pip install -e '.[test]'

pytest                            # everything; L3 skips unless its assets are present
pytest -m l0                      # pure functions
pytest -m l1                      # model structural contracts
pytest -m l2                      # end-to-end on the synthetic fixture
pytest -m l3                      # accuracy; skips without gated assets
pytest -m "l0 or l1 or l2"        # exactly what CI runs
```

Markers are declared in `pyproject.toml` and enforced with `--strict-markers`, so a
typo in a marker name is a collection error rather than a silently skipped test.

Regenerating the fixtures:

```bash
python tests/data/make_random_init.py    # rewrite the untrained checkpoint
python -m tests.data.synthetic_movie     # print a summary of the synthetic movie
```

## Why L3 does not run in CI

L3 needs **both** gated assets, and neither is available to an automated runner:

1. **Trained weights** are gated on the HuggingFace `vanvalenlab` organization — access
   requires org membership and accepting the license terms. They are never committed;
   `VISION.md` constraint 6. The pinned revision and hash are in `metrics/PROVENANCE.md`.
2. **The dataset** has no public distribution route at all right now. Model hosting has
   moved to HuggingFace but the datasets have not, and the `users.deepcell.org` portal
   that previously issued data API keys was removed. See `doc/model_access.md`.

Both L3 fixtures call `pytest.skip` rather than failing, so a plain `pytest` run is
green without them. Running L3 at all currently requires a local copy of the dataset.

## Where the assertions are

Most layers are scaffolding right now, with few or no assertions. That is intentional.

The behavioral assertions arrive with the per-capability `characterize-*` changes, next
to the written requirements they verify, so that a failing test always points at a
specification rather than at someone's recollection of how the code used to behave.
The planning artifacts are under `openspec/`.

What L2 asserts today is narrower than it will be: that tracking is **deterministic**,
and that its assignment decisions are **not near ties**. The second is a precondition
for the first being useful — frozen golden outputs are only reproducible across
platforms while the linear assignment decisions sit clear of ties. The smallest observed
margin on the fixture is `1.0e-03`, and the guard fails below `1e-6`.

## What a green run does and does not mean

L2 runs the model with **untrained** weights, so it is blind to anything the model
learned. It detects changes in the tracking decision path — ingestion, embedding, the
cost matrix, the assignment, division detection, lineage construction — and nothing
about accuracy.

**A green suite is not evidence that the model still works.** Accuracy is L3's job,
measured against the baseline recorded in `metrics/PROVENANCE.md`.
