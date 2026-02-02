# Copyright 2016-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tracking/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for evaluating tracking performance"""

from collections import Counter
import itertools
import functools

from skimage.measure import regionprops

import numpy as np

from geometry_utils import compute_overlap_vectorized
import pandas as pd
import networkx as nx

def match_nodes(gt, res, threshold=1):
    """Relabel predicted track to match GT track labels.

    Args:
        gt (np arr): label movie (y) from ground truth .trk file.
        res (np arr): label movie (y) from predicted results .trk file
        threshold (optional, float): threshold value for IoU to count as same cell. Default 1.
            If segmentations are identical, 1 works well.
            For imperfect segmentations try 0.6-0.8 to get better matching

    Returns:
        gtcells (np arr): Array of overlapping ids in the gt movie.
        rescells (np arr): Array of overlapping ids in the res movie.

    Raises:
        ValueError: If .
    """
    num_frames = gt.shape[0]
    iou = np.zeros((num_frames, np.max(gt) + 1, np.max(res) + 1))

    # TODO: Compute IOUs only when neccesary
    # If bboxs for true and pred do not overlap with each other, the assignment is immediate
    # Otherwise use pixel-wise IOU to determine which cell is which

    # Regionprops expects one frame at a time
    for frame in range(num_frames):
        gt_frame = gt[frame]
        res_frame = res[frame]

        gt_props = regionprops(np.squeeze(gt_frame.astype('int')))
        gt_boxes = [np.array(gt_prop.bbox) for gt_prop in gt_props]
        gt_boxes = np.array(gt_boxes).astype('double')
        gt_box_labels = [int(gt_prop.label) for gt_prop in gt_props]

        res_props = regionprops(np.squeeze(res_frame.astype('int')))
        res_boxes = [np.array(res_prop.bbox) for res_prop in res_props]
        res_boxes = np.array(res_boxes).astype('double')
        res_box_labels = [int(res_prop.label) for res_prop in res_props]

        overlaps = compute_overlap_vectorized(gt_boxes, res_boxes)    # has the form [gt_bbox, res_bbox]

        # Find the bboxes that have overlap at all (ind_ corresponds to box number - starting at 0)
        ind_gt, ind_res = np.nonzero(overlaps)

        # frame_ious = np.zeros(overlaps.shape)
        for index in range(ind_gt.shape[0]):

            iou_gt_idx = gt_box_labels[ind_gt[index]]
            iou_res_idx = res_box_labels[ind_res[index]]
            intersection = np.logical_and(gt_frame == iou_gt_idx, res_frame == iou_res_idx)
            union = np.logical_or(gt_frame == iou_gt_idx, res_frame == iou_res_idx)
            iou[frame, iou_gt_idx, iou_res_idx] = intersection.sum() / union.sum()

    gtcells, rescells = np.where(np.nansum(iou, axis=0) >= threshold)

    return gtcells, rescells


def trk_to_graph(lineage, node_key=None):
    """Converts a lineage dictionary into a graph representation of the lineages

    Args:
        lineage (dict): Dictionary of lineage data
        node_key (dict): Map between gt nodes and result nodes

    Returns:
        networkx.Graph: Graph representation of the lineage data.
    """
    edges = []

    all_ids = set()
    single_nodes = set()
    attributes = {}

    for i, lin in lineage.items():
        # Update cell id if node_key is available
        if node_key and (i in node_key):
            idx = node_key[i]
        else:
            idx = i

        cellids = ['{}_{}'.format(idx, t) for t in lin['frames']]

        if len(cellids) == 1:
            single_nodes.add(cellids[0])

        all_ids.update(cellids)
        edges.append(pd.DataFrame({
            'source': cellids[0:-1],
            'target': cellids[1:]
        }))

        # Add connections to any daughters
        source = '{}_{}'.format(idx, max(lin['frames']))
        for d in lin['daughters']:
            # Update cell id if node_key is available
            if node_key and (i in node_key):
                d_idx = node_key[d]
            else:
                d_idx = d

            # Assume daughter appears in next frame
            target = '{}_{}'.format(d_idx, max(lin['frames']) + 1)
            edges.append(pd.DataFrame({
                'source': [source],
                'target': [target]
            }))

            attributes[source] = {'division': True}

    # Create graph
    edges = pd.concat(edges)
    G = nx.from_pandas_edgelist(edges, source='source', target='target', create_using=nx.DiGraph)
    nx.set_node_attributes(G, attributes)

    # Add all isolates to graph
    for cell_id in single_nodes:
        G.add_node(cell_id)

    return G


def map_node(gt_node, G_res, cells_gt, cells_res):
    """Finds the res node that matches the gt_node submitted

    Args:
        gt_node (str): String matching form '{cell id}_{frame}'
        G_res (networkx.graph): Graph of the results
        cells_gt (np.array): Array containing ground truth cell ids corresponding to res ids
        cells_res (np.array): Array containing corresponding res ids
    """
    idx = int(gt_node.split('_')[0])
    frame = int(gt_node.split('_')[1])

    if idx in cells_gt:
        for r_idx in cells_res[cells_gt == idx]:
            # Check if node exists with the right frame
            r_node = '{}_{}'.format(r_idx, frame)
            if r_node in G_res.nodes:
                return r_node
        else:
            # Can't find result node so return original gt node
            return gt_node
    elif gt_node in G_res.nodes:
        return gt_node
    else:
        return gt_node


def classify_divisions(G_gt, G_res, cells_gt=[], cells_res=[]):
    """Compare two graphs and calculate the cell division confusion matrix.

    WARNING: This function will only work if the labels underlying both
    graphs are the same. E.G. the parents only match if the same label
    splits in the same frame - but each movie isn't guaranteed to be labeled
    in the same way (with the same order). Should be used with match_nodes

    Args:
        G_gt (networkx.Graph): Ground truth cell lineage graph.
        G_res (networkx.Graph): Predicted cell lineage graph.
        cells_gt (np.ndarray): List of ground truth cell ids from `match_nodes`
        cells_res (np.ndarray): List of result cell ids from `match_nodes`

    Returns:
        dict: Diciontary of all division statistics

    Raises:
        ValueError: cells_gt and cells_res must be the same length
    """
    if len(cells_gt) != len(cells_res):
        raise ValueError('cells_gt and cells_res must be the same length.')

    def _map_node(gt_node):
        return map_node(gt_node, G_res, cells_gt, cells_res)

    # Identify nodes with parent attribute
    div_gt = [node for node, d in G_gt.nodes(data=True)
              if d.get('division', False)]
    div_res = [node for node, d in G_res.nodes(data=True)
               if d.get('division', False)]

    correct = []         # Correct division
    incorrect = []       # Wrong/mismatch division
    missed = []          # Missed division

    for node in div_gt:
        idx = int(node.split('_')[0])
        frame = int(node.split('_')[1])

        # Check if the index is mapped onto a different results index
        if idx in cells_gt:
            for r_idx in cells_res[cells_gt == idx]:
                # Check if node exists with the right frame
                r_node = '{}_{}'.format(r_idx, frame)
                if r_node in G_res.nodes:
                    break  # Exit for loop since we found the right node
            else:
                # Node doesn't exist so count this division as missed
                print('missed node {} division completely'.format(node))
                missed.append(node)
                continue  # move on to next node in div_gt
        # Check if the node exists with same id in G_res
        elif node in G_res.nodes:
            r_node = node
        # Node doesn't exist
        else:
            print('missed node {} division completely'.format(node))
            missed.append(node)
            continue  # move on to next node in div_gt

        # If we found the results node, evaluate division result
        # Get gt predecessors and successors for comparsion
        # Map gt nodes onto results nodes if possible
        pred_gt = [_map_node(n) for n in G_gt.pred[node]]
        succ_gt = [_map_node(n) for n in G_gt.succ[node]]

        # Check if res node was also called a division
        if r_node in div_res:
            # Get res predecessors and successor
            pred_res = list(G_res.pred[r_node])
            succ_res = list(G_res.succ[r_node])

            # Parents and daughters are the same, perfect!
            if (Counter(pred_gt) == Counter(pred_res) and
                    Counter(succ_gt) == Counter(succ_res)):
                correct.append(node)

            else:  # what went wrong?
                incorrect.append(node)
                errors = ['out degree = {}'.format(G_res.out_degree(r_node))]
                if Counter(succ_gt) != Counter(succ_res):
                    errors.append('daughters mismatch')
                if Counter(pred_gt) != Counter(pred_res):
                    errors.append('parents mismatch')
                if G_res.out_degree(r_node) == G_gt.out_degree(node):
                    errors.append('gt and res degree equal')
                print(node, '{}.'.format(', '.join(errors)))

            div_res.remove(r_node)

        else:  # valid division not in results, it was missed
            print('missed node {} division completely'.format(node))
            missed.append(node)

    # Count any remaining res nodes as false positives
    false_positive = div_res

    return {
        'correct_division': correct,
        'mismatch_division': incorrect,
        'false_positive_division': false_positive,
        'false_negative_division': missed,
        'total_divisions': len(div_gt)
    }


def correct_shifted_divisions(
        false_negative_division, false_positive_division, correct_division,
        y_gt, y_res,
        G_gt, G_res,
        threshold):
    """Correct divisions errors that are shifted by a frame and should be counted as correct

    Args:
        false_negative_division (list): List of nodes classifed as a false negative division
        false_positive_division (list): List of nodes classified as false positive division
        correct_division (list): List of nodes where divisions were correctly assigned
        y_gt (np.array): Y mask for the ground truth data
        y_res (np.array): Y mask for the predicted data
        G_gt (networkx.graph): Graph of the ground truth
        G_res (networkx.graph): Graph of the results
        threshold (float): Value between 0 and 1 used to determine matching cells using IoU

    Returns:
        dict: Dictionary of updated false_negative_division, false_positive_division
            and correct_division lists
    """

    metrics = {
        'false_negative_division': false_negative_division,
        'false_positive_division': false_positive_division,
        'correct_division': correct_division
    }
    y = {'gt': y_gt, 'res': y_res}
    G = {'gt': G_gt, 'res': G_res}

    # Explicitly label nodes according to source
    false_negative_division = ['gt-' + n for n in false_negative_division]
    false_positive_division = ['res-' + n for n in false_positive_division]

    # Convert to dictionary for lookup by frame
    d_false_negative_division, d_fp = {}, {}
    for d, j in [(d_false_negative_division, false_negative_division),
                 (d_fp, false_positive_division)]:
        for n in j:
            t = int(n.split('_')[-1])
            v = d.get(t, [])
            v.append(n)
            d[t] = v

    frame_pairs = []
    for t in d_false_negative_division:
        if t + 1 in d_fp:
            frame_pairs.append((t, t + 1))
        if t - 1 in d_fp:
            frame_pairs.append((t - 1, t))

    # Convert to set to remove any duplicates
    frame_pairs = list(set(frame_pairs))

    matches = []

    # Loop over each pair of frames
    for t1, t2 in frame_pairs:
        # Get nodes from each frames
        n1s = d_false_negative_division.get(t1, []) + d_fp.get(t1, [])
        n2s = d_false_negative_division.get(t2, []) + d_fp.get(t2, [])

        # Compare each pair and save if they are above the threshold
        for n1, n2 in itertools.product(n1s, n2s):
            source1, node1 = n1.split('-')[0], n1.split('-')[1]
            source2, node2 = n2.split('-')[0], n2.split('-')[1]

            # Check if the nodes are from different sources
            if source1 == source2:
                continue

            # Compare sum of daughters in n1 to parent in n2
            daughters = [int(d.split('_')[0]) for d in list(G[source1].succ[node1])]
            if len(daughters) == 1:
                mask1 = y[source1][t2] == daughters[0]
            else:
                mask1 = np.logical_or(
                    y[source1][t2] == daughters[0],
                    y[source1][t2] == daughters[1])
                if len(daughters) > 2:
                    for d in range(2, len(daughters)):
                        mask1 = np.logical_or(
                            mask1,
                            y[source1][t2] == daughters[d]
                        )
            mask2 = y[source2][t2] == int(node2.split('_')[0])

            # Compute iou
            intersection = np.logical_and(mask1, mask2)
            union = np.logical_or(mask1, mask2)
            iou = intersection.sum() / union.sum()
            if iou >= threshold:
                matches.extend([n1, n2])

    # Remove matches from the list of errors
    for n in matches:
        source, node = n.split('-')[0], n.split('-')[1]
        # Remove error counts
        if source == 'gt':
            metrics['false_negative_division'].remove(node)
            # Add node to the correct_division count
            metrics['correct_division'].append(node)
            print('corrected division {} as a frameshift division not an error'.format(node))
        elif source == 'res':
            metrics['false_positive_division'].remove(node)

    return metrics


def calculate_association_accuracy(lineage_gt, lineage_res, cells_gt=[], cells_res=[]):
    """Calculate the association accuracy for each ground truth lineage

    Defined as the number of true positive associations between cells divided by
    the total number of ground truth associations. Associations are equivalent to
    the edges that connect cells in a graph. As described by:
        - Hayashida, J., Nishimura, K., and Bise, R. (2020). MPM: Joint
          Representation of Motion and Position Map for Cell Tracking. In 2020 IEEE/CVF
          Conference on Computer Vision and Pattern Recognition (CVPR) (IEEE).
        - Nishimura, K., Hayashida, J., Wang, C., Ker, D.F.E., and Bise, R. (2020).
          Weakly-Supervised Cell Tracking via Backward-and-Forward Propagation. In
          Computer Vision - ECCV 2020 Lecture Notes in Computer Science.

    Args:
        lineage_gt (dict): Ground truth lineages
        linage_res (dict): Predicted lineages
        cells_gt (list): List of ground truth cell ids from `match_nodes`
        cells_res (list): List of result cell ids from `match_nodes`

    Returns:
        int: Number of true positive associations
        int: Total number of associations

    Raises:
        ValueError: cells_gt and cells_res must be the same length
    """
    if len(cells_gt) != len(cells_res):
        raise ValueError('cells_gt and cells_res must be the same length.')

    true_positive = 0
    total = 0

    for g_idx, g_lin in lineage_gt.items():
        # Calculate gt edges
        g_frames = g_lin['frames']
        g_edges = ['{}-{}'.format(t0, t1) for t0, t1 in zip(g_frames[:-1], g_frames[1:])]
        total += len(g_edges)

        # Check for any mappings
        if g_idx in cells_gt:
            scores = []
            for r_idx in cells_res[cells_gt == g_idx]:
                r_frames = lineage_res[r_idx]['frames']
                r_edges = ['{}-{}'.format(t0, t1) for t0, t1 in zip(r_frames[:-1], r_frames[1:])]
                scores.append(sum(r in g_edges for r in r_edges))
            true_positive += max(scores)

        # Check if the idx already matches
        elif g_idx in lineage_res:
            r_frames = lineage_res[g_idx]['frames']
            r_edges = ['{}-{}'.format(t0, t1) for t0, t1 in zip(r_frames[:-1], r_frames[1:])]
            true_positive += sum(r in g_edges for r in r_edges)

    return true_positive, total


def calculate_target_effectiveness(lineage_gt, lineage_res, cells_gt=[], cells_res=[]):
    """Calculate the target effectiveness. Final score can be obtained by dividing
    true_positive by total

    The TE measure considers the number of cell instances correctly associated within
    a track with respect to the total number of cells in a track. Only the best possible
    true positive score is recorded for each ground truth lineage As described by:
        - Hayashida, J., Nishimura, K., and Bise, R. (2020). MPM: Joint
          Representation of Motion and Position Map for Cell Tracking. In 2020 IEEE/CVF
          Conference on Computer Vision and Pattern Recognition (CVPR) (IEEE).
        - Nishimura, K., Hayashida, J., Wang, C., Ker, D.F.E., and Bise, R. (2020).
          Weakly-Supervised Cell Tracking via Backward-and-Forward Propagation. In
          Computer Vision - ECCV 2020 Lecture Notes in Computer Science.

    Args:
        lineage_gt (dict): Ground truth lineages
        linage_res (dict): Predicted lineages
        cells_gt (list): List of ground truth cell ids from `match_nodes`
        cells_res (list): List of result cell ids from `match_nodes`

    Returns:
        int: Number of true positive assignments of cells to lineages
        int: Number of cells present in ground truth

    Raises:
        ValueError: cells_gt and cells_res must be the same length
    """
    if len(cells_gt) != len(cells_res):
        raise ValueError('cells_gt and cells_res must be the same length.')

    true_positive = 0
    total = 0

    for g_idx, g_lin in lineage_gt.items():
        # Check for any mappings
        if g_idx in cells_gt:
            # Collect candidates for overlaps, but only save the best
            scores = []
            for r_idx in cells_res[cells_gt == g_idx]:
                r_frames = lineage_res[r_idx]['frames']
                scores.append(sum(r in g_lin['frames'] for r in r_frames))

            true_positive += max(scores)

        # Check if the idx already matches
        elif g_idx in lineage_res:
            r_frames = lineage_res[g_idx]['frames']
            true_positive += sum(r in g_lin['frames'] for r in r_frames)

        # Save total assigments for this gt lineage
        total += len(g_lin['frames'])

    return true_positive, total


def calculate_summary_stats(correct_division,
                            false_positive_division,
                            false_negative_division,
                            total_divisions,
                            aa_total, aa_tp,
                            te_total, te_tp,
                            n_digits=2):
    """Calculate additional summary statistics for tracking performance
    based on results of classify_divisions

    Catch ZeroDivisionError and set to 0 instead

    Args:
        correct_division (int): True positive or "correct divisions"
        false_positive_division (int): False positives
        false_negative_division (int): False negatives
        total_divisions (int): Total number of ground truth divisions
        aa_total (int): Total number of ground truth associations
        aa_tp (int): True positive associations
        te_total (int): Total number of target assignments
        te_tp (int): True positive target assignments
        n_digits (int, optional): Number of digits to round to. Default 2.
    """

    _round = functools.partial(round, ndigits=n_digits)

    try:
        recall = correct_division / (correct_division + false_negative_division)
    except ZeroDivisionError:
        recall = 0

    try:
        precision = correct_division / (correct_division + false_positive_division)
    except ZeroDivisionError:
        precision = 0

    try:
        f1 = 2 * (recall * precision) / (recall + precision)
    except ZeroDivisionError:
        f1 = 0

    try:
        mbc = correct_division / (correct_division
                                  + false_negative_division
                                  + false_positive_division)
    except ZeroDivisionError:
        mbc = 0

    try:
        fraction_miss = false_negative_division / total_divisions
    except ZeroDivisionError:
        fraction_miss = 0

    try:
        aa = aa_tp / aa_total
    except ZeroDivisionError:
        aa = 0

    try:
        te = te_tp / te_total
    except ZeroDivisionError:
        te = 0

    return {
        'Division Recall': _round(recall),
        'Division Precision': _round(precision),
        'Division F1': _round(f1),
        'Mitotic branching correctness': _round(mbc),
        'Fraction missed divisions': _round(fraction_miss),
        'Association Accuracy': _round(aa),
        'Target Effectiveness': _round(te)
    }


class TrackingMetrics:
    def __init__(self,
                 lineage_gt, y_gt,
                 lineage_res, y_res,
                 threshold=1,
                 allow_division_shift=True):
        """Class to coordinate the benchmarking of a pair of trk files

        Args:
            lineage_gt (dict): Ground truth lineages
            linage_res (dict): Predicted lineages
            y_gt (np.array): Y mask for the ground truth data
            y_res (np.array): Y mask for the predicted data
            threshold (optional, float): threshold value for IoU to count as same cell. Default 1.
                If segmentations are identical, 1 works well.
                For imperfect segmentations try 0.6-0.8 to get better matching
            allow_division_shift (optional, bool): Allows divisions to be treated as correct if
                they are off by a single frame. Default True.
        """

        self.lineage_gt = lineage_gt
        self.lineage_res = lineage_res
        self.y_gt = y_gt
        self.y_res = y_res
        self.threshold = threshold
        self.allow_division_shift = allow_division_shift

        # Match up labels in GT to Results to allow for direct comparisons
        self.cells_gt, self.cells_res = match_nodes(y_gt, y_res, self.threshold)

        # Generate graphs without remapping nodes to avoid losing lineages
        self.G_gt = trk_to_graph(lineage_gt)
        self.G_res = trk_to_graph(lineage_res)

        self.stats = self.calculate_metrics()

    def calculate_metrics(self):
        # Classify divison errors
        stats = classify_divisions(
            self.G_gt, self.G_res, cells_gt=self.cells_gt, cells_res=self.cells_res)

        if self.allow_division_shift:
            updates = correct_shifted_divisions(
                false_negative_division=stats['false_negative_division'],
                false_positive_division=stats['false_positive_division'],
                correct_division=stats['correct_division'],
                y_gt=self.y_gt,
                y_res=self.y_res,
                G_gt=self.G_gt,
                G_res=self.G_res,
                threshold=self.threshold)

            for k, v in updates.items():
                stats[k] = v

        # Convert list of nodes to counts
        for k, v in stats.items():
            if isinstance(v, list):
                stats[k] = len(v)

        # Calculate aa and te
        aa_tp, aa_total = calculate_association_accuracy(
            self.lineage_gt, self.lineage_res, self.cells_gt, self.cells_res)

        te_tp, te_total = calculate_target_effectiveness(
            self.lineage_gt, self.lineage_res, self.cells_gt, self.cells_res)

        return {
            **stats,
            'aa_tp': aa_tp,
            'aa_total': aa_total,
            'te_tp': te_tp,
            'te_total': te_total
        }


def benchmark_tracking_performance(trk_gt, trk_res, threshold=1, allow_division_shift=True):
    """Compare two related .trk files (one being the GT of the other)

    Calculate division statistics, target effectiveness and association accuracy

    Currently included for backwards compatibility, but is no longer necessary

    Args:
        trk_gt (path): Path to the ground truth .trk file.
        trk_res (path): Path to the predicted results .trk file.
        threshold (optional, float): threshold value for IoU to count as same cell. Default 1.
            If segmentations are identical, 1 works well.
            For imperfect segmentations try 0.6-0.8 to get better matching
    """

    # Load data
    m = TrackingMetrics.from_trk_files(trk_gt, trk_res,
                                       threshold=threshold,
                                       allow_division_shift=allow_division_shift)

    return m.stats
