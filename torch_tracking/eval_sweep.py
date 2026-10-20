"""Inference and evaluation scripts for GNN cell tracking model"""

import pandas as pd
import numpy as np
from torch_tracking.tracker import CellTracker
import zarr
import json
import tqdm
import itertools
from pathlib import Path

from torch_tracking.metrics import TrackingMetrics

def build_indices(X):
    """Find the last populated (non-zero) frame index for each batch element.

    Parameters
    ----------
    X : numpy.ndarray
        Batch of time-lapse label/image arrays with shape
        ``(B, T, H, W, C)``, where trailing all-zero frames indicate
        padding beyond the true movie length.

    Returns
    -------
    list of int
        For each batch element, the index of the last non-zero frame.
    """
    samples = []

    for batch in range(X.shape[0]):

        end_frame = np.sum(np.sum(X[batch], axis=(1, 2, 3)) != 0) - 1
        samples.append(end_frame.item())
    
    return samples


if __name__ == "__main__":

    config = {
            'batch_size': 6,
            'n_layers': 2,
            'crop_size': 32,
            'crop_mode': 'fixed'
        }

    # Initialize model

    metrics_out = Path('metrics')

    if not metrics_out.exists():
        metrics_out.mkdir()

    checkpoint_dir = Path.home() / '.deepcell/models/tracking/best_model.pt'

    z = zarr.open(Path.home() / '.deepcell/tracking/test.zarr')

    with open(Path.home() / '.deepcell/tracking/test.json') as file:
        gt_lineage = json.load(file)

    division_sweep = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    birth_sweep = [0.999]
    death_sweep = [0.999]

    metrics_out = []

    X = z['X'][:]
    y = z['y'][:]

    all_comb = [tup for tup in itertools.product(*[division_sweep, birth_sweep, death_sweep])]

    for div_thresh, birth_thresh, death_thresh in tqdm.tqdm(all_comb):
        for batch in tqdm.tqdm(range(X.shape[0]), leave=False):

            curr_gt_lineage= gt_lineage[batch]

            samples = build_indices(y)
            end_frame = samples[batch]

            tracker = CellTracker(
                checkpoint_dir=checkpoint_dir,
                device='cuda:1',
                division=div_thresh,
                birth=birth_thresh,
                death=death_thresh,
                track_length=8,
                verbose=False
            )

            tracker.preprocess_movie(movie=X[batch, :end_frame],
                                    annotation=y[batch, :end_frame])

            tracker.track_cells()

            y_tracked = tracker.y_tracked
            lineage = tracker.get_lineage_dict()

            curr_gt_lineage  = {int(k): v for k, v in curr_gt_lineage.items()}
            lineage = {int(k): v for k, v in lineage.items()}

            metrics = TrackingMetrics(curr_gt_lineage, y[batch, :end_frame].squeeze(), lineage, y_tracked.squeeze(), threshold=0.8, verbose=False).stats
            metrics['div'] = div_thresh
            metrics['birth'] = birth_thresh
            metrics['death'] = death_thresh
            metrics['set_id'] = batch+1

            metrics_out.append(metrics)

    df = pd.DataFrame(metrics_out)

    df['division_precision'] = (df['correct_division'])/(df['correct_division'] + df['false_positive_division'])
    df['division_recall'] = df['correct_division']/(df['correct_division'] + df['false_negative_division'])
    df['division_f1'] = (2 * df['division_recall'] * df['division_precision'])/(df['division_precision'] + df['division_recall'])

    df['aa_accuracy'] = df['aa_tp']/df['aa_total']
    df['te_accuracy'] = df['te_tp']/df['te_total']
    
    df.to_csv('metrics/postprocess_sweep.csv')

