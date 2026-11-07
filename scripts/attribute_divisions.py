"""Attribute every ground-truth division to the stage that lost it.

This is the measurement behind `openspec/changes/correct-checkpoint-selection/design.md`
(Decision 1). It answers: of the true divisions in a split, how many did the tracker lose
before its division detection ever looked at them, and how many did detection see but turn
down?

The tracker and the ground-truth lineage share one label space (the raw annotation labels),
so each true division (parent P at frame f, daughters at f+1) can be followed through the
tracker's own decisions without any overlap matching. Every true daughter edge lands in
exactly one bucket, in the order the pipeline applies them:

  GATE     - the daughter was never offered to division detection at all: the assignment
             step continued it onto an existing track, so no parent was sought for it.
  CAPPED   - the true parent was a candidate but skipped because it had already divided.
  SIBLING  - the true parent was skipped by the newly-appeared-sibling guard.
  LOWPROB  - the true parent was a live candidate but its daughter probability did not
             clear the division threshold.
  OK       - the daughter probability cleared the threshold on the true parent.

A division is *detectable* when at least one of its daughters is OK: one tested daughter is
enough, because the sibling is recovered from the parent's own continuation.

CPU only. About five minutes per pass over a 27-movie split. Example:

    OMP_NUM_THREADS=4 nice -n 10 python scripts/attribute_divisions.py \
        --split-dir ~/.deepcell/tracking/regen --split val \
        --checkpoint checkpoints/<run>/best_model.pt --out attribution.json
"""

import argparse
import collections
import json
from pathlib import Path

DEFAULT_DIVISION = 0.5

BUCKETS = ('OK', 'GATE', 'LOWPROB', 'CAPPED', 'SIBLING')


def build_label_map(y):
    """(frame, original label) -> the globally unique id the tracker gives that cell.

    `clean_up_annotations` numbers every cell of every frame afresh, in ascending label
    order, frame by frame, starting at total_cells + 1.
    """
    import numpy as np

    uniques = []
    for frame in range(y.shape[0]):
        labels = np.unique(y[frame])
        uniques.append(labels[labels != 0])
    next_id = sum(len(u) for u in uniques) + 1
    label_map = {}
    for frame, labels in enumerate(uniques):
        for label in labels:
            label_map[(frame, int(label))] = next_id
            next_id += 1
    return label_map


def attribute_movie(tracker, X, y, lineage, division=DEFAULT_DIVISION):
    """Track one movie and attribute each of its true divisions.

    Returns (records, stats): one record per true division, each listing its daughters'
    buckets, and the harness's own division counts for the same tracker run.
    """
    from torch_tracking.metrics import TrackingMetrics

    end = y.shape[0]
    gt = {int(k): v for k, v in lineage.items()}

    # Log every `_get_parent` call and every candidate parent it weighed.
    calls = {}
    original = tracker._get_parent

    def logged(frame, cell_id, predictions):
        candidates = []
        for row, track_id in enumerate(predictions['track_ids']):
            track = tracker.tracks[track_id]
            for cell_idx in range(predictions['predictions'].shape[1]):
                if tracker.idx_to_id[(frame, cell_idx)] == cell_id:
                    candidates.append({
                        'labels': [int(x) for x in track['frame_labels']],
                        'capped': bool(track['capped']),
                        'sibling': track['frames'] == [frame],
                        'prob': float(predictions['predictions'][row, cell_idx, 2]),
                    })
        calls[(int(frame), int(cell_id))] = candidates
        return original(frame, cell_id, predictions)

    tracker._get_parent = logged
    label_map = build_label_map(y)
    tracker.preprocess_movie(movie=X, annotation=y)
    tracker.track_cells()

    stats = TrackingMetrics(gt, y.squeeze(), tracker.get_lineage_dict(),
                            tracker.y_tracked.squeeze(), threshold=0.8, verbose=False).stats

    records = []
    for parent, parent_data in gt.items():
        daughters = parent_data.get('daughters') or []
        frames = [f for f in (parent_data.get('frames') or []) if f < end]
        if not daughters or not frames:
            continue
        parent_clean = label_map.get((max(frames), int(parent)))
        record = {'parent': int(parent), 'n_daughters': len(daughters), 'daughters': []}
        for daughter in daughters:
            daughter_frames = [f for f in (gt.get(int(daughter), {}).get('frames') or []) if f < end]
            if not daughter_frames:
                record['daughters'].append({'label': int(daughter), 'bucket': 'OUT_OF_RANGE'})
                continue
            first = min(daughter_frames)
            candidates = calls.get((first, label_map.get((first, int(daughter)))))
            if candidates is None:
                record['daughters'].append({'label': int(daughter), 'bucket': 'GATE'})
                continue
            # The parent's track may already have reached the daughters' frame before this
            # birth was processed, so match on the track's whole label history.
            true_parent = [c for c in candidates if parent_clean in c['labels']]
            if not true_parent:
                record['daughters'].append({'label': int(daughter), 'bucket': 'NO_PARENT_TRACK'})
                continue
            best = max(true_parent, key=lambda c: c['prob'])
            if best['capped']:
                bucket = 'CAPPED'
            elif best['sibling']:
                bucket = 'SIBLING'
            elif best['prob'] > division:
                bucket = 'OK'
            else:
                bucket = 'LOWPROB'
            record['daughters'].append({'label': int(daughter), 'bucket': bucket,
                                        'prob': best['prob']})
        records.append(record)
    return records, stats


def detectable(record):
    """Whether detection could have recorded this division: one daughter cleared is enough."""
    return any(d['bucket'] == 'OK' for d in record['daughters'])


def check_instrument_agrees_with_metric(records, stats):
    """The check the attribution rests on, for one movie.

    A division the harness scores correct must be one this instrument calls detectable,
    since the tracker can only record a division through a daughter it tested and cleared.
    Returns (n_correct_by_metric, n_detectable). The metric's count can never exceed the
    instrument's; where a movie scores every division correct the two must be equal.
    """
    n_detectable = sum(detectable(r) for r in records)
    return stats['correct_division'], n_detectable


def summarize(records):
    """Edge counts per bucket, and division counts, over a set of records."""
    edges = collections.Counter(d['bucket'] for r in records for d in r['daughters'])
    total = sum(edges.values())
    lines = [f'divisions: {len(records)}   daughter edges: {total}']
    for bucket in (*BUCKETS, *sorted(set(edges) - set(BUCKETS))):
        if edges[bucket]:
            lines.append(f'  {bucket:<16}{edges[bucket]:>6}  {100 * edges[bucket] / total:5.1f}%')
    lines.append(f'detectable divisions: {sum(detectable(r) for r in records)}')
    return edges, '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--split-dir', required=True, help='directory holding <split>.zarr and <split>.json')
    parser.add_argument('--split', default='val')
    parser.add_argument('--checkpoint', required=True, help='a trained checkpoint (.pt)')
    parser.add_argument('--division', type=float, default=DEFAULT_DIVISION)
    parser.add_argument('--threads', type=int, default=4, help='CPU threads; this is a shared machine')
    parser.add_argument('--movies', help='comma-separated movie indices; default is every movie')
    parser.add_argument('--out', help='write the per-division records here as JSON')
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    import zarr
    from torch_tracking.eval import build_indices
    from torch_tracking.tracker import CellTracker
    from torch_tracking.utils import DISTANCE_THRESHOLD, require_mpp

    split_dir = Path(args.split_dir).expanduser()
    z = zarr.open(str(split_dir / f'{args.split}.zarr'))
    mpps = require_mpp(z, str(split_dir / f'{args.split}.zarr'))
    lineages = json.load(open(split_dir / f'{args.split}.json'))
    end_frames = build_indices(z['y'][:])
    movies = ([int(m) for m in args.movies.split(',')] if args.movies
              else range(z['X'].shape[0]))

    all_records, movie_stats = [], []
    for movie in movies:
        end = int(end_frames[movie])
        tracker = CellTracker(checkpoint_dir=args.checkpoint, device='cpu', mpp=float(mpps[movie]),
                              division=args.division, birth=0.999, death=0.999, track_length=8,
                              distance_threshold=DISTANCE_THRESHOLD, verbose=False)
        records, stats = attribute_movie(tracker, z['X'][movie, :end], z['y'][movie, :end],
                                         lineages[movie], division=args.division)
        correct, n_detectable = check_instrument_agrees_with_metric(records, stats)
        print(f'movie {movie}: divisions={len(records)} correct_by_metric={correct} '
              f'detectable={n_detectable}', flush=True)
        if correct > n_detectable:
            raise SystemExit(f'movie {movie}: the metric scored {correct} divisions correct but '
                             f'only {n_detectable} are detectable; the attribution is unsound')
        for record in records:
            record['movie'] = movie
        all_records.extend(records)
        movie_stats.append({'movie': movie, 'correct_division': correct,
                            'total_divisions': len(records), 'detectable': n_detectable})

    print(summarize(all_records)[1])
    if args.out:
        Path(args.out).write_text(json.dumps({'records': all_records, 'movies': movie_stats}, indent=1))
        print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
