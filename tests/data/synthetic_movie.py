"""A procedurally generated nuclear time-lapse for end-to-end (L2) tests.

Why synthetic rather than a real excerpt: trained weights are gated on HuggingFace and
there is currently no route for a new user to obtain DynamicNuclearNet-tracking at all
(see doc/model_access.md). A fixture built from either would confine end-to-end testing
to one workstation. This movie derives from no dataset, so L2 runs anywhere.

What it is NOT: a measure of accuracy. Tracking this movie with untrained weights says
nothing about whether the model is any good - that is L3's job. L2 is a change-detector
for the tracking decision path: ingestion, embedding, the cost matrix, the LAP, division
detection, and lineage construction.

The movie contains four nuclei drifting linearly, one of which divides. Motion is slow
and nuclei stay well separated, so assignments are unambiguous.

Layout (labels are 1-based; 0 is background):

    frames 0-4      frames 5-9
    ----------      ----------
    1  drifting     1  drifting
    2  drifting     2  drifting
    3  drifting     3  drifting
    4  drifting     4 divides -> 5 and 6 appear, moving apart

Label 4 ends at frame 4; labels 5 and 6 begin at frame 5. That is the .trk convention -
a parent terminates and two new labels begin - and it satisfies `is_valid_lineage`.
"""

from dataclasses import dataclass

import numpy as np
from skimage.draw import ellipse

#: Frame on which label 4 divides into labels 5 and 6.
DIVISION_FRAME = 5
#: Parent label and its two daughters.
PARENT_LABEL = 4
DAUGHTER_LABELS = (5, 6)

N_FRAMES = 10
HEIGHT = 160
WIDTH = 160
RADIUS_Y = 9
RADIUS_X = 7


@dataclass(frozen=True)
class SyntheticMovie:
    """A synthetic movie and its ground-truth lineage.

    Attributes:
        X: ``(T, H, W, 1)`` float32 image.
        y: ``(T, H, W, 1)`` int32 label mask; 0 is background.
        lineage: ground-truth lineage keyed by **str** label, matching the ``.trk``
            format and what ``is_valid_lineage`` expects. Note that
            ``CellTracker.get_lineage_dict()`` returns **int** keys instead - that
            mismatch is recorded as divergence 1 and belongs to ``lineage-output``.
    """

    X: np.ndarray
    y: np.ndarray
    lineage: dict


def _draw(X, y, frame, label, cy, cx):
    rr, cc = ellipse(int(round(cy)), int(round(cx)), RADIUS_Y, RADIUS_X, shape=(HEIGHT, WIDTH))
    y[frame, rr, cc, 0] = label
    # A deterministic, label-dependent intensity so the appearance encoder sees
    # something other than a constant field.
    X[frame, rr, cc, 0] = 0.6 + 0.3 * np.sin(label + frame)


def make_synthetic_movie() -> SyntheticMovie:
    """Build the fixture. Deterministic: no RNG is involved anywhere."""
    X = np.zeros((N_FRAMES, HEIGHT, WIDTH, 1), dtype=np.float32)
    y = np.zeros((N_FRAMES, HEIGHT, WIDTH, 1), dtype=np.int32)

    for f in range(N_FRAMES):
        _draw(X, y, f, 1, 30 + 1.5 * f, 30 + 1.0 * f)
        _draw(X, y, f, 2, 110 - 1.0 * f, 40 + 1.5 * f)
        _draw(X, y, f, 3, 40 + 0.8 * f, 120 - 1.2 * f)
        if f < DIVISION_FRAME:
            _draw(X, y, f, PARENT_LABEL, 120, 120 + 0.5 * f)
        else:
            d = f - DIVISION_FRAME
            _draw(X, y, f, DAUGHTER_LABELS[0], 118 - 1.2 * d, 122 + 1.2 * d)
            _draw(X, y, f, DAUGHTER_LABELS[1], 126 + 1.2 * d, 118 - 1.2 * d)

    all_frames = list(range(N_FRAMES))
    lineage = {
        "1": {"label": 1, "frames": all_frames, "parent": None, "daughters": []},
        "2": {"label": 2, "frames": all_frames, "parent": None, "daughters": []},
        "3": {"label": 3, "frames": all_frames, "parent": None, "daughters": []},
        str(PARENT_LABEL): {
            "label": PARENT_LABEL,
            "frames": list(range(DIVISION_FRAME)),
            "parent": None,
            "daughters": list(DAUGHTER_LABELS),
        },
    }
    for d in DAUGHTER_LABELS:
        lineage[str(d)] = {
            "label": d,
            "frames": list(range(DIVISION_FRAME, N_FRAMES)),
            "parent": PARENT_LABEL,
            "daughters": [],
        }

    return SyntheticMovie(X=X, y=y, lineage=lineage)


if __name__ == "__main__":
    m = make_synthetic_movie()
    print("X", m.X.shape, m.X.dtype, "| y", m.y.shape, m.y.dtype)
    print("cells per frame:", [int(len(np.unique(m.y[t])) - 1) for t in range(N_FRAMES)])
    print("lineage:", {k: v["frames"][:1] + ["..."] + v["frames"][-1:] for k, v in m.lineage.items()})
