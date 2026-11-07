"""Measure what a looser division gate would actually be worth.

This is the measurement behind `openspec/changes/correct-division-scoring/design.md`
(Decision 4) and divergence 78. `scripts/attribute_divisions.py` showed that many true
daughters are never offered to division detection (the "gate"), and bounded the gain by an
oracle that assumes every gated edge becomes reachable at no cost. This script measures a
rule that can actually be written, by running the tracker with the rule switched on and
scoring the lineages it produces with the project's own metric.

Two rules, both applied on top of the unchanged tracker, leaving the birth gate alone:

  sibling  A track that did not divide, and that the assignment step continued onto cell c,
           is treated as having divided if c AND at least one other cell in the same frame
           each clear `threshold` for its daughter probability (class 2). The other cells
           become its daughters (the tracker's own post-pass turns c into the remaining
           daughter). A cell already linked to a different track is taken from that track.
  any      The same, but one other cell clearing the threshold is enough and the
           continuation need not clear it. This is "admit every cell to division
           detection", not just births.

Baseline (no rule) and every rule are run over the same movies, and each is scored with
`TrackingMetrics`. Per rule it reports, pooled over the split:

  recovers      = baseline missed - rule missed       (true divisions no longer missed)
  daughter set  = rule correct - baseline correct     (net gain in exactly-right divisions)
  false firings = rule false positive - baseline false positive
  delta F1      = rule F1 - baseline F1, under the corrected scoring rule (a mismatch counts
                  as both a miss and a false positive) and under the previous rule

It also reports the shape of the baseline's mismatched divisions: how many have only one of
two true daughters (the gate defect caught one stage later), an extra daughter, or a wrong
one.

NOTE: this is a re-implementation. The probe that first produced the `correct-division-scoring`
design's tables was lost, so the design now quotes this script's numbers instead. Its baseline
matches the recorded validation counts (444 divisions, 14 mismatched, 20 missed, corrected-rule
F1 0.9393, 7 mismatches with one of two daughters). A differently worded rule would give
different gains: the figures are this rule's, not a bound on every possible rule.

Reproduce the design's table (validation split, baseline checkpoint, GPU 5 only, about
20 minutes on one GPU; sets threads because this is a shared machine):

    CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 nice -n 10 python scripts/measure_sibling_rule.py \
        --split-dir ~/.deepcell/tracking/regen --split val \
        --checkpoint checkpoints/20261106-051755/best_model.pt --device cuda \
        --out sibling_rule.json

Sanity check before trusting a run: the baseline line must reproduce the recorded validation
counts (444 divisions, 14 mismatched, 20 missed, pooled corrected-rule F1 0.9393).
"""

import argparse
import json
from pathlib import Path

RULES = (
    ('sibling', 0.90),
    ('sibling', 0.95),
    ('sibling', 0.98),
    ('sibling', 0.50),
    ('any', 0.50),
)

COUNTS = ('correct_division', 'mismatch_division',
          'false_positive_division', 'false_negative_division')


def make_tracker_class():
    from torch_tracking.tracker import CellTracker

    class RuleTracker(CellTracker):
        """CellTracker with an optional extra rule for declaring a division."""

        rule = None          # None, 'sibling' or 'any'
        rule_threshold = 0.95

        def _update_tracks(self, assignments, frame, predictions):
            self._forced = {}
            if self.rule is not None:
                assignments = self._apply_rule(assignments, frame, predictions)
            return super()._update_tracks(assignments, frame, predictions)

        def _get_parent(self, frame, cell_id, predictions):
            forced = getattr(self, '_forced', {})
            if cell_id in forced:
                return forced[cell_id]
            return super()._get_parent(frame, cell_id, predictions)

        def _apply_rule(self, assignments, frame, predictions):
            preds = predictions['predictions']
            track_ids = predictions['track_ids']
            cells = self._get_cells_in_frame(frame)
            n_tracks = len(track_ids)
            n_cells = len(cells)

            continuation = {int(r): int(c) for r, c in assignments
                            if r < n_tracks and c < n_cells}

            claims = {}   # cell index -> (probability, parent track)
            for row, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                if track['capped'] or track['frames'] == [frame]:
                    continue
                probs = preds[row, :n_cells, 2]
                cleared = [j for j in range(n_cells) if probs[j] > self.rule_threshold]
                cont = continuation.get(row)
                others = [j for j in cleared if j != cont]
                if self.rule == 'sibling':
                    fires = cont in cleared and len(others) >= 1
                else:
                    fires = len(others) >= 1
                if not fires:
                    continue
                for j in others:
                    if j not in claims or probs[j] > claims[j][0]:
                        claims[j] = (float(probs[j]), track_id)

            if not claims:
                return assignments

            rewritten = assignments.copy()
            for k, (row, cell_idx) in enumerate(assignments):
                if int(cell_idx) in claims and row < n_tracks \
                        and track_ids[row] != claims[int(cell_idx)][1]:
                    rewritten[k, 0] = n_tracks   # any row >= n_tracks is a birth
            for cell_idx, (_, parent) in claims.items():
                self._forced[cells[cell_idx]] = parent
            return rewritten

    return RuleTracker


def score(tracker, X, y, lineage):
    """Track one movie and return the harness's own division counts."""
    from torch_tracking.metrics import TrackingMetrics

    gt = {int(k): v for k, v in lineage.items()}
    tracker.preprocess_movie(movie=X, annotation=y)
    tracker.track_cells()
    metrics = TrackingMetrics(gt, y.squeeze(), tracker.get_lineage_dict(),
                              tracker.y_tracked.squeeze(), threshold=0.8, verbose=False)
    return metrics, {k: int(metrics.stats[k]) for k in (*COUNTS, 'total_divisions')}


def mismatch_shapes(metrics):
    """Shape of each mismatched division of one movie, by result out-degree."""
    from torch_tracking.metrics import classify_divisions, map_node

    out = classify_divisions(metrics.G_gt, metrics.G_res, cells_gt=metrics.cells_gt,
                             cells_res=metrics.cells_res, verbose=False)
    shapes = {'one_of_two_daughters': 0, 'extra_daughter': 0, 'wrong_daughter': 0}
    for node in out['mismatch_division']:
        r_node = map_node(node, metrics.G_res, metrics.cells_gt, metrics.cells_res)
        gt_degree = metrics.G_gt.out_degree(node)
        res_degree = metrics.G_res.out_degree(r_node)
        if res_degree < gt_degree:
            shapes['one_of_two_daughters'] += 1
        elif res_degree > gt_degree:
            shapes['extra_daughter'] += 1
        else:
            shapes['wrong_daughter'] += 1
    return shapes


def pooled_f1(counts, corrected=True):
    c, m = counts['correct_division'], counts['mismatch_division']
    fp, fn = counts['false_positive_division'], counts['false_negative_division']
    extra = m if corrected else 0
    precision = c / (c + fp + extra) if c + fp + extra else 0.0
    recall = c / (c + fn + extra) if c + fn + extra else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def add(a, b):
    return {k: a.get(k, 0) + b[k] for k in b}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu', help="'cuda' sees only CUDA_VISIBLE_DEVICES")
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--movies', help='comma-separated movie indices; default is every movie')
    parser.add_argument('--out')
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    import zarr
    from torch_tracking.eval import build_indices
    from torch_tracking.utils import DISTANCE_THRESHOLD, require_mpp

    Tracker = make_tracker_class()
    split_dir = Path(args.split_dir).expanduser()
    z = zarr.open(str(split_dir / f'{args.split}.zarr'))
    mpps = require_mpp(z, str(split_dir / f'{args.split}.zarr'))
    lineages = json.load(open(split_dir / f'{args.split}.json'))
    end_frames = build_indices(z['y'][:])
    movies = ([int(m) for m in args.movies.split(',')] if args.movies
              else range(z['X'].shape[0]))

    variants = [('baseline', None, None)] + [(f'{r} {t:.2f}', r, t) for r, t in RULES]
    totals = {name: {} for name, _, _ in variants}
    shapes = {'one_of_two_daughters': 0, 'extra_daughter': 0, 'wrong_daughter': 0}

    for movie in movies:
        end = int(end_frames[movie])
        X, y = z['X'][movie, :end], z['y'][movie, :end]
        for name, rule, threshold in variants:
            tracker = Tracker(checkpoint_dir=args.checkpoint, device=args.device,
                              mpp=float(mpps[movie]), division=0.5, birth=0.999, death=0.999,
                              track_length=8, distance_threshold=DISTANCE_THRESHOLD,
                              verbose=False)
            tracker.rule, tracker.rule_threshold = rule, threshold or 0.95
            metrics, counts = score(tracker, X, y, lineages[movie])
            totals[name] = add(totals[name], counts)
            if rule is None:
                for k, v in mismatch_shapes(metrics).items():
                    shapes[k] += v
        base = totals['baseline']
        print(f'movie {movie}: done (baseline so far: {base["correct_division"]} correct, '
              f'{base["mismatch_division"]} mismatched, {base["false_negative_division"]} missed '
              f'of {base["total_divisions"]})', flush=True)

    base = totals['baseline']
    f1_new, f1_old = pooled_f1(base), pooled_f1(base, corrected=False)
    print()
    print(f'baseline: {base}')
    print(f'baseline pooled division F1: {f1_new:.4f} corrected rule, {f1_old:.4f} previous rule')
    print(f'baseline mismatch shapes: {shapes}')
    print()
    print(f'{"rule":<14}{"recovers":>9}{"set right":>10}{"false":>7}{"dF1 corr":>10}{"dF1 prev":>10}')
    rows = []
    for name, rule, _ in variants[1:]:
        t = totals[name]
        recovers = base['false_negative_division'] - t['false_negative_division']
        right = t['correct_division'] - base['correct_division']
        false = t['false_positive_division'] - base['false_positive_division']
        d_new, d_old = pooled_f1(t) - f1_new, pooled_f1(t, corrected=False) - f1_old
        rows.append({'rule': name, 'recovers': recovers, 'daughter_set_right': right,
                     'false_firings': false, 'delta_f1_corrected': d_new,
                     'delta_f1_previous': d_old, 'counts': t})
        print(f'{name:<14}{recovers:>9}{right:>10}{false:>7}{d_new:>+10.4f}{d_old:>+10.4f}')

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'baseline': base, 'baseline_f1_corrected': f1_new, 'baseline_f1_previous': f1_old,
             'mismatch_shapes': shapes, 'rules': rows}, indent=1))
        print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
