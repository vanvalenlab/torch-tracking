"""Parameter sweep over postprocessing thresholds, using the shared evaluation rules.

This is a different program from :mod:`torch_tracking.eval` - it produces one row per
(movie, parameter combination) rather than one per movie - so it keeps its own entry
point. What it does not keep is its own copy of the frame-extent rule: that is imported
from :mod:`torch_tracking.eval` so both harnesses evaluate the same frames.

Importing this module performs no sweep.

It also shares the other evaluation rules: every non-empty frame is evaluated, and each
movie is tracked at the resolution its zarr declares.
"""

from pathlib import Path
import itertools
import json

import pandas as pd
import tqdm
import zarr

from torch_tracking.eval import build_indices, _resolve_split
from torch_tracking.metrics import TrackingMetrics
from torch_tracking.tracker import CellTracker
from torch_tracking.utils import require_mpp, DISTANCE_THRESHOLD

__all__ = ['sweep', 'build_indices']

DEFAULT_DIVISION_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DEFAULT_BIRTH_SWEEP = (0.999,)
DEFAULT_DEATH_SWEEP = (0.999,)


def sweep(
    checkpoint_path,
    split_dir,
    split='test',
    output_dir='metrics',
    device='cpu',
    division_sweep=DEFAULT_DIVISION_SWEEP,
    birth_sweep=DEFAULT_BIRTH_SWEEP,
    death_sweep=DEFAULT_DEATH_SWEEP,
    track_length=8,
    distance_threshold=DISTANCE_THRESHOLD,
    threshold=0.8,
    verbose=False,
    progress=True,
):
    """Evaluate a checkpoint across a grid of postprocessing thresholds.

    Args and behavior mirror :func:`torch_tracking.eval.evaluate`, except that the
    results carry one row per (movie, parameter combination) and are written to
    ``postprocess_sweep.csv``.

    Returns:
        pandas.DataFrame: the per-combination results.

    Raises:
        FileNotFoundError: if the split's zarr or lineage JSON is absent.
    """
    zarr_path, lineage_path = _resolve_split(split_dir, split)

    z = zarr.open(str(zarr_path))
    mpps = require_mpp(z, zarr_path)
    with open(lineage_path) as handle:
        gt_lineages = json.load(handle)

    X = z['X'][:]
    y = z['y'][:]

    # The shared rule, imported rather than redefined.
    end_frames = build_indices(y)

    rows = []

    combinations = list(itertools.product(division_sweep, birth_sweep, death_sweep))
    if progress:
        combinations = tqdm.tqdm(combinations)

    for div_thresh, birth_thresh, death_thresh in combinations:
        batches = range(X.shape[0])
        if progress:
            batches = tqdm.tqdm(batches, leave=False)

        for batch in batches:
            end_frame = end_frames[batch]

            tracker = CellTracker(
                checkpoint_path=checkpoint_path,
                device=device,
                mpp=float(mpps[batch]),
                division=div_thresh,
                birth=birth_thresh,
                death=death_thresh,
                track_length=track_length,
                distance_threshold=distance_threshold,
                verbose=verbose,
            )

            tracker.preprocess_movie(
                movie=X[batch, :end_frame],
                annotation=y[batch, :end_frame],
            )

            tracker.track_cells()

            y_tracked = tracker.y_tracked
            lineage = tracker.get_lineage_dict()  # integer-keyed already
            gt_lineage = {int(k): v for k, v in gt_lineages[batch].items()}

            stats = TrackingMetrics(
                gt_lineage,
                y[batch, :end_frame].squeeze(),
                lineage,
                y_tracked.squeeze(),
                threshold=threshold,
                verbose=verbose,
            ).stats

            stats['div'] = div_thresh
            stats['birth'] = birth_thresh
            stats['death'] = death_thresh
            stats['set_id'] = batch + 1

            rows.append(stats)

    df = pd.DataFrame(rows)

    df['division_precision'] = df['correct_division'] / (
        df['correct_division'] + df['false_positive_division'] + df['mismatch_division'])
    df['division_recall'] = df['correct_division'] / (
        df['correct_division'] + df['false_negative_division'] + df['mismatch_division'])
    df['division_f1'] = (2 * df['division_recall'] * df['division_precision']) / (df['division_precision'] + df['division_recall'])

    df['aa_accuracy'] = df['aa_tp'] / df['aa_total']
    df['te_accuracy'] = df['te_tp'] / df['te_total']

    if output_dir is not None:
        # As in eval.evaluate: the directory created is the directory written into.
        results_dir = Path(output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_dir / 'postprocess_sweep.csv')

    return df


def _cli(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split-dir', required=True,
                        help='Directory holding <split>.zarr and <split>.json')
    parser.add_argument('--split', default='test')
    parser.add_argument('--output-dir', default='metrics')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--division', type=float, nargs='+', default=list(DEFAULT_DIVISION_SWEEP))
    parser.add_argument('--birth', type=float, nargs='+', default=list(DEFAULT_BIRTH_SWEEP))
    parser.add_argument('--death', type=float, nargs='+', default=list(DEFAULT_DEATH_SWEEP))
    parser.add_argument('--track-length', type=int, default=8)
    parser.add_argument('--distance-threshold', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.8)

    args = parser.parse_args(argv)

    return sweep(
        checkpoint_path=args.checkpoint,
        split_dir=args.split_dir,
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
        division_sweep=tuple(args.division),
        birth_sweep=tuple(args.birth),
        death_sweep=tuple(args.death),
        track_length=args.track_length,
        distance_threshold=args.distance_threshold,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    _cli()
