"""Inference and evaluation harness for the GNN cell tracking model.

Importing this module performs no evaluation: it loads no checkpoint, reads no
dataset and writes no file. Call :func:`evaluate` to run an evaluation.

Every non-empty frame of each movie is evaluated (see :func:`build_indices`), and each
movie is tracked at the resolution it declares: that resolution is read from the split's
zarr, so the tracker rescales by ``mpp / MODEL_MPP`` exactly as conversion did.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tqdm
import zarr

from torch_tracking.tracker import CellTracker
from torch_tracking.utils import require_mpp, count_nonempty_frames, DISTANCE_THRESHOLD
from torch_tracking.metrics import TrackingMetrics

__all__ = [
    'build_indices',
    'evaluate',
    'pooled_metrics',
    'pretty_print',
    'EvaluationResult',
    'create_timelapse_gif',
]

#: Columns carrying the counts every pooled figure is derived from. The pooled
#: metrics are recomputable from these alone, which is what makes the written CSV
#: an authoritative record rather than a rendering.
COUNT_COLUMNS = (
    'correct_division',
    'false_positive_division',
    'false_negative_division',
    'aa_tp',
    'aa_total',
    'te_tp',
    'te_total',
)


def build_indices(y):
    """Per-movie frame extent: the count of non-empty mask frames.

    This is the single definition of the rule; ``eval_sweep`` imports it rather than
    carrying its own copy, and the training sampler derives its extent from the same
    ``count_nonempty_frames``. The value is the exclusive end of the movie's occupied
    frames, so it is used as a slice bound that includes every non-empty frame.
    """
    return [count_nonempty_frames(y[batch]) for batch in range(y.shape[0])]


def pooled_metrics(df):
    """Pool the contract metrics across movies by summing counts, not averaging.

    Each metric's numerator and denominator are summed across every movie and
    divided once. A movie whose own ratio is undefined - no divisions to find, say -
    still contributes its counts, because only the count columns are consulted.

    Averaging per-movie ratios instead gives a different and wrong answer: on the
    recorded test split, mean-of-ratios association accuracy is 0.9715 where the
    pooled value is 0.9728.
    """
    totals = {column: float(df[column].sum()) for column in COUNT_COLUMNS}

    correct = totals['correct_division']
    false_positive = totals['false_positive_division']
    false_negative = totals['false_negative_division']

    predicted = correct + false_positive
    actual = correct + false_negative

    precision = correct / predicted if predicted else float('nan')
    recall = correct / actual if actual else float('nan')

    if precision + recall:
        division_f1 = 2 * precision * recall / (precision + recall)
    else:
        division_f1 = float('nan')

    aa = totals['aa_tp'] / totals['aa_total'] if totals['aa_total'] else float('nan')
    te = totals['te_tp'] / totals['te_total'] if totals['te_total'] else float('nan')

    return {
        'division_precision': precision,
        'division_recall': recall,
        'division_f1': division_f1,
        'aa_accuracy': aa,
        'te_accuracy': te,
    }


@dataclass
class EvaluationResult:
    """What an evaluation produced, returned to the caller as well as written.

    Attributes:
        per_movie: One row per evaluated movie, carrying the counts in
            ``COUNT_COLUMNS`` plus the per-movie ratio columns.
        pooled: Pooled division F1, association accuracy and target
            effectiveness, plus the pooled division precision and recall.
        results_path: Where the per-movie CSV was written, or ``None`` when no
            output directory was given.
    """

    per_movie: pd.DataFrame
    pooled: dict = field(default_factory=dict)
    results_path: Path = None


def _resolve_split(split_dir, split):
    """Locate a split's zarr and lineage JSON, failing with the path that is missing."""
    split_dir = Path(split_dir)

    zarr_path = split_dir / f'{split}.zarr'
    lineage_path = split_dir / f'{split}.json'

    for path in (zarr_path, lineage_path):
        if not path.exists():
            raise FileNotFoundError(
                f'Evaluation split not found: {path} does not exist. '
                f'Pass split_dir= and split= pointing at a directory holding '
                f'{split}.zarr and {split}.json.'
            )

    return zarr_path, lineage_path


def evaluate(
    checkpoint_path,
    split_dir,
    split='test',
    output_dir='metrics',
    device='cpu',
    division=0.5,
    birth=0.999,
    death=0.999,
    track_length=8,
    distance_threshold=DISTANCE_THRESHOLD,
    threshold=0.8,
    write_movies=False,
    movies_dir='movies',
    verbose=False,
    progress=True,
):
    """Evaluate a checkpoint against a ground-truth split.

    Args:
        checkpoint_path: The checkpoint to evaluate. Required; no path is assumed.
        split_dir: Directory holding ``<split>.zarr`` and ``<split>.json``.
        split: Split name within ``split_dir``. Defaults to ``'test'``.
        output_dir: Directory the per-movie CSV is written beneath. Pass ``None``
            to compute without writing. Defaults to ``'metrics'``.
        device: Compute device. Defaults to CPU; no GPU is required.
        division, birth, death, track_length, distance_threshold: Tracking
            parameters, forwarded to :class:`CellTracker`.
        threshold: Match threshold for :class:`TrackingMetrics`.
        write_movies: Write per-movie comparison GIFs beneath ``movies_dir``.
        movies_dir: Where those GIFs go, created only when ``write_movies``.
        verbose: Passed through to the tracker.
        progress: Show a progress bar.

    Returns:
        EvaluationResult: the per-movie table and the pooled figures.

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

    # Loop-invariant: the extents depend only on the masks.
    end_frames = build_indices(y)

    rows = []

    batches = range(X.shape[0])
    if progress:
        batches = tqdm.tqdm(batches, leave=False)

    for batch in batches:
        end_frame = end_frames[batch]

        # A tracker is constructed per movie, so no state carries across movies
        # and evaluation order cannot affect any movie's result.
        # It is given the movie's declared resolution, so it is rescaled to MODEL_MPP.
        tracker = CellTracker(
            checkpoint_dir=checkpoint_path,
            device=device,
            mpp=float(mpps[batch]),
            division=division,
            birth=birth,
            death=death,
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

        rows.append(stats)

        if write_movies:
            movies_out = Path(movies_dir)
            movies_out.mkdir(parents=True, exist_ok=True)
            create_timelapse_gif(
                tracker._track_review_dict()['y_tracked'],
                y[batch, :end_frame],
                output_path=str(movies_out / f'timelapse_batch_{batch}.gif'),
                cmap='viridis',
            )

    df = pd.DataFrame(rows)

    df['division_precision'] = df['correct_division'] / (df['correct_division'] + df['false_positive_division'])
    df['division_recall'] = df['correct_division'] / (df['correct_division'] + df['false_negative_division'])
    df['division_f1'] = (2 * df['division_recall'] * df['division_precision']) / (df['division_precision'] + df['division_recall'])

    df['aa_accuracy'] = df['aa_tp'] / df['aa_total']
    df['te_accuracy'] = df['te_tp'] / df['te_total']

    pooled = pooled_metrics(df)

    results_path = None
    if output_dir is not None:
        # The directory created is the directory written into. An earlier version
        # bound this name twice and wrote to the process working directory instead.
        results_dir = Path(output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / 'eval_results.csv'
        df.to_csv(results_path)

    return EvaluationResult(per_movie=df, pooled=pooled, results_path=results_path)


def pretty_print(pooled):
    """Print every contract metric, so none needs aggregating outside the harness."""
    print()
    print('=' * 25)
    print('Results')
    print('=' * 25)

    labels = (
        ('Division precision', 'division_precision'),
        ('Division recall', 'division_recall'),
        ('Division F1', 'division_f1'),
        ('Association accuracy', 'aa_accuracy'),
        ('Target effectiveness', 'te_accuracy'),
    )

    for label, key in labels:
        print(f'{label}: {pooled[key]:.4}')

    print()


def create_timelapse_gif(im1, im2, output_path='timelapse.gif', fps=10,
                         titles=('Predicted', 'True'),
                         cmap='gray'):
    """Create a GIF comparing two time-lapse movies side by side.

    Parameters:
    -----------
    im1, im2 : numpy.ndarray
        Time lapse images of shape (T, H, W, C)
    output_path : str
        Path to save the output GIF
    fps : int
        Frames per second for the GIF
    titles : tuple
        Titles for the two subplots
    cmap : str
        Colormap to use for display
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    T = im1.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im1_plot = axes[0].imshow(im1[0], cmap=cmap)
    im2_plot = axes[1].imshow(im2[0], cmap=cmap)

    axes[0].set_title(titles[0])
    axes[1].set_title(titles[1])
    axes[0].axis('off')
    axes[1].axis('off')

    plt.colorbar(im1_plot, ax=axes[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2_plot, ax=axes[1], fraction=0.046, pad=0.04)

    frame_text = fig.text(0.5, 0.02, f'Frame: 0/{T - 1}', ha='center', fontsize=12)

    plt.tight_layout()

    def update(frame):
        im1_plot.set_data(im1[frame])
        im2_plot.set_data(im2[frame])
        frame_text.set_text(f'Frame: {frame}/{T - 1}')
        return im1_plot, im2_plot, frame_text

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=True)
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()

    print(f'GIF saved to {output_path}')


def _cli(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', required=True, help='Checkpoint to evaluate')
    parser.add_argument('--split-dir', required=True,
                        help='Directory holding <split>.zarr and <split>.json')
    parser.add_argument('--split', default='test')
    parser.add_argument('--output-dir', default='metrics')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--division', type=float, default=0.5)
    parser.add_argument('--birth', type=float, default=0.999)
    parser.add_argument('--death', type=float, default=0.999)
    parser.add_argument('--track-length', type=int, default=8)
    parser.add_argument('--distance-threshold', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--write-movies', action='store_true')
    parser.add_argument('--movies-dir', default='movies')

    args = parser.parse_args(argv)

    result = evaluate(
        checkpoint_path=args.checkpoint,
        split_dir=args.split_dir,
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
        division=args.division,
        birth=args.birth,
        death=args.death,
        track_length=args.track_length,
        distance_threshold=args.distance_threshold,
        threshold=args.threshold,
        write_movies=args.write_movies,
        movies_dir=args.movies_dir,
    )

    pretty_print(result.pooled)
    if result.results_path is not None:
        print(f'Wrote {result.results_path}')

    return result


if __name__ == '__main__':
    _cli()
