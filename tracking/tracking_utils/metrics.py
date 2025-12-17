"""
Evaluation metrics for cell tracking with GNN models.

Provides multiple evaluation strategies:
1. Cost-based scoring (as requested)
2. TRA-style tracking accuracy (recommended)
3. Per-event metrics (divisions, merges, splits)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class TrackingEvaluator:
    """
    Comprehensive evaluator for cell tracking results.
    
    Takes predicted and ground truth temporal adjacency matrices and computes
    various tracking quality metrics.
    
    Args:
        penalty_weights: Dictionary of penalty weights for different error types.
            Default weights:
            - 'fp_track': 1.0 (false positive - track that shouldn't exist)
            - 'fn_track': 1.0 (false negative - missed track)
            - 'swap': 0.5 (track ID swap between two tracks)
            - 'missed_division': 2.0 (failed to detect mitosis)
            - 'false_division': 2.0 (incorrectly predicted mitosis)
            - 'wrong_parent': 1.5 (daughter assigned to wrong parent)
    """
    
    def __init__(self, penalty_weights: Optional[Dict[str, float]] = None):
        
        # Default penalty weights
        default_weights = {
            'fp_track': 1.0,        # False positive track
            'fn_track': 1.0,        # False negative track  
            'swap': 0.5,            # Track ID swap
            'missed_division': 2.0,  # Missed mitosis
            'false_division': 2.0,   # False mitosis detection
            'wrong_parent': 1.5,     # Wrong parent assignment
            'gap': 0.3,             # Small temporal gap in track
        }
        
        self.weights = penalty_weights if penalty_weights else default_weights
        
    def _extract_lineage_from_adjacency(
        self, 
        temporal_adj: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Extract lineage structure from temporal adjacency matrix.
        
        Args:
            temporal_adj: (T-1, N, N, 3) adjacency matrix where:
                [:, :, :, 0] = same cell probability
                [:, :, :, 1] = different cell probability  
                [:, :, :, 2] = daughter cell probability
        
        Returns:
            Dictionary mapping track_id -> {
                'frames': list of frames where track appears,
                'daughters': list of daughter track IDs,
                'parent': parent track ID or None
            }
        """
        
        T_minus_1, N, _, _ = temporal_adj.shape
        T = T_minus_1 + 1
        
        lineage = {}
        
        # Initialize all possible tracks
        for track_id in range(N):
            lineage[track_id] = {
                'frames': [],
                'daughters': [],
                'parent': None
            }
        
        # Process each frame transition
        for t in range(T_minus_1):
            frame_adj = temporal_adj[t]  # (N, N, 3)
            
            # Check for same cell links (diagonal should be high)
            for track_id in range(N):
                same_prob = frame_adj[track_id, track_id, 0]
                
                # If same cell link exists, add frame to track
                if same_prob > 0.5:  # threshold
                    if t not in lineage[track_id]['frames']:
                        lineage[track_id]['frames'].append(t)
                    if t + 1 not in lineage[track_id]['frames']:
                        lineage[track_id]['frames'].append(t + 1)
            
            # Check for division events
            for parent_id in range(N):
                for daughter_id in range(N):
                    if parent_id == daughter_id:
                        continue
                    
                    division_prob = frame_adj[parent_id, daughter_id, 2]
                    
                    if division_prob > 0.5:  # threshold
                        lineage[parent_id]['daughters'].append(daughter_id)
                        lineage[daughter_id]['parent'] = parent_id
        
        # Clean up - remove tracks with no frames
        lineage = {k: v for k, v in lineage.items() if len(v['frames']) > 0}
        
        return lineage
    
    def _match_tracks(
        self,
        pred_lineage: Dict[int, Dict],
        gt_lineage: Dict[int, Dict]
    ) -> Tuple[Dict[int, int], set, set]:
        """
        Match predicted tracks to ground truth tracks using IoU overlap.
        
        Args:
            pred_lineage: Predicted lineage structure
            gt_lineage: Ground truth lineage structure
        
        Returns:
            - matches: Dict mapping pred_track_id -> gt_track_id
            - unmatched_pred: Set of unmatched predicted track IDs
            - unmatched_gt: Set of unmatched ground truth track IDs
        """
        
        pred_ids = list(pred_lineage.keys())
        gt_ids = list(gt_lineage.keys())
        
        if not pred_ids or not gt_ids:
            return {}, set(pred_ids), set(gt_ids)
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
        
        for i, pred_id in enumerate(pred_ids):
            pred_frames = set(pred_lineage[pred_id]['frames'])
            
            for j, gt_id in enumerate(gt_ids):
                gt_frames = set(gt_lineage[gt_id]['frames'])
                
                intersection = len(pred_frames & gt_frames)
                union = len(pred_frames | gt_frames)
                
                if union > 0:
                    iou_matrix[i, j] = intersection / union
        
        # Use Hungarian algorithm to find optimal matching
        # Maximize IoU = minimize negative IoU
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        matches = {}
        matched_pred = set()
        matched_gt = set()
        
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] > 0.5:  # Threshold for valid match
                pred_id = pred_ids[i]
                gt_id = gt_ids[j]
                matches[pred_id] = gt_id
                matched_pred.add(pred_id)
                matched_gt.add(gt_id)
        
        unmatched_pred = set(pred_ids) - matched_pred
        unmatched_gt = set(gt_ids) - matched_gt
        
        return matches, unmatched_pred, unmatched_gt
    
    def compute_cost_based_score(
        self,
        pred_temporal_adj: np.ndarray,
        gt_temporal_adj: np.ndarray,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Compute cost-based tracking score as requested.
        
        This starts from a perfect score and subtracts penalties for each error.
        
        Args:
            pred_temporal_adj: (T-1, N, N, 3) predicted adjacency
            gt_temporal_adj: (T-1, N, N, 3) ground truth adjacency
            normalize: If True, normalize score by number of ground truth tracks
        
        Returns:
            Dictionary with:
                - 'total_score': Final score after penalties
                - 'max_score': Maximum possible score
                - 'normalized_score': Score normalized to [0, 1]
                - 'error_counts': Dictionary of error type counts
                - 'error_costs': Dictionary of costs per error type
        """
        
        # Extract lineages
        pred_lineage = self._extract_lineage_from_adjacency(pred_temporal_adj)
        gt_lineage = self._extract_lineage_from_adjacency(gt_temporal_adj)
        
        # Match tracks
        matches, unmatched_pred, unmatched_gt = self._match_tracks(
            pred_lineage, gt_lineage
        )
        
        # Initialize error tracking
        error_counts = defaultdict(int)
        error_costs = defaultdict(float)
        
        # Start with perfect score (1 point per ground truth track)
        max_score = len(gt_lineage)
        current_score = float(max_score)
        
        # 1. Penalize false positive tracks (predicted but not in GT)
        error_counts['fp_track'] = len(unmatched_pred)
        error_costs['fp_track'] = error_counts['fp_track'] * self.weights['fp_track']
        current_score -= error_costs['fp_track']
        
        # 2. Penalize false negative tracks (in GT but not predicted)
        error_counts['fn_track'] = len(unmatched_gt)
        error_costs['fn_track'] = error_counts['fn_track'] * self.weights['fn_track']
        current_score -= error_costs['fn_track']
        
        # 3. For matched tracks, check for errors
        for pred_id, gt_id in matches.items():
            pred_track = pred_lineage[pred_id]
            gt_track = gt_lineage[gt_id]
            
            # Check for temporal gaps
            pred_frames = set(pred_track['frames'])
            gt_frames = set(gt_track['frames'])
            
            missed_frames = gt_frames - pred_frames
            extra_frames = pred_frames - gt_frames
            
            gap_penalty = (len(missed_frames) + len(extra_frames)) * self.weights['gap']
            error_counts['gap'] += len(missed_frames) + len(extra_frames)
            error_costs['gap'] += gap_penalty
            current_score -= gap_penalty
            
            # Check division events
            pred_daughters = set(pred_track['daughters'])
            gt_daughters = set(gt_track['daughters'])
            
            # Missed divisions
            missed_divisions = len(gt_daughters - pred_daughters)
            error_counts['missed_division'] += missed_divisions
            error_costs['missed_division'] += missed_divisions * self.weights['missed_division']
            current_score -= missed_divisions * self.weights['missed_division']
            
            # False divisions
            false_divisions = len(pred_daughters - gt_daughters)
            error_counts['false_division'] += false_divisions
            error_costs['false_division'] += false_divisions * self.weights['false_division']
            current_score -= false_divisions * self.weights['false_division']
            
            # Wrong parent assignments for matched daughters
            for pred_daughter in pred_daughters:
                if pred_daughter in matches:
                    gt_daughter = matches[pred_daughter]
                    if gt_daughter in gt_daughters:
                        # Correct daughter detection
                        pass
                    else:
                        # Daughter assigned to wrong parent
                        error_counts['wrong_parent'] += 1
                        error_costs['wrong_parent'] += self.weights['wrong_parent']
                        current_score -= self.weights['wrong_parent']
        
        # Check for track swaps (more sophisticated)
        # Two tracks that should be separate but got swapped
        swap_count = self._detect_swaps(matches, pred_lineage, gt_lineage)
        error_counts['swap'] = swap_count
        error_costs['swap'] = swap_count * self.weights['swap']
        current_score -= error_costs['swap']
        
        # Normalize if requested
        normalized_score = current_score / max_score if max_score > 0 else 0.0
        normalized_score = max(0.0, min(1.0, normalized_score))  # Clamp to [0, 1]
        
        return {
            'total_score': current_score,
            'max_score': max_score,
            'normalized_score': normalized_score,
            'error_counts': dict(error_counts),
            'error_costs': dict(error_costs)
        }
    
    def _detect_swaps(
        self,
        matches: Dict[int, int],
        pred_lineage: Dict[int, Dict],
        gt_lineage: Dict[int, Dict]
    ) -> int:
        """
        Detect track ID swaps.
        
        A swap occurs when two predicted tracks map to two GT tracks,
        but their frame assignments are crossed.
        
        This is a simplified heuristic - real swap detection is complex.
        """
        
        swap_count = 0
        matched_pairs = list(matches.items())
        
        for i in range(len(matched_pairs)):
            for j in range(i + 1, len(matched_pairs)):
                pred_i, gt_i = matched_pairs[i]
                pred_j, gt_j = matched_pairs[j]
                
                # Get frame sets
                pred_i_frames = set(pred_lineage[pred_i]['frames'])
                pred_j_frames = set(pred_lineage[pred_j]['frames'])
                gt_i_frames = set(gt_lineage[gt_i]['frames'])
                gt_j_frames = set(gt_lineage[gt_j]['frames'])
                
                # Check for swap pattern:
                # pred_i should overlap with gt_i, but also overlaps significantly with gt_j
                # AND pred_j should overlap with gt_j, but also overlaps significantly with gt_i
                
                correct_overlap_i = len(pred_i_frames & gt_i_frames)
                cross_overlap_i_j = len(pred_i_frames & gt_j_frames)
                
                correct_overlap_j = len(pred_j_frames & gt_j_frames)
                cross_overlap_j_i = len(pred_j_frames & gt_i_frames)
                
                # Only count as swap if cross-overlap is substantial AND
                # cross-overlap exceeds correct overlap (indicating actual swap)
                if (cross_overlap_i_j > 2 and cross_overlap_j_i > 2 and
                    cross_overlap_i_j > correct_overlap_i and 
                    cross_overlap_j_i > correct_overlap_j):
                    swap_count += 1
        
        return swap_count
    
    def compute_tra_score(
        self,
        pred_temporal_adj: np.ndarray,
        gt_temporal_adj: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute TRA (Tracking Accuracy) score.
        
        TRA is a standard metric from the Cell Tracking Challenge.
        
        TRA = 1 - min(1, (AOGM / AOGM_0))
        
        where AOGM is the Acyclic Oriented Graph Matching score.
        
        Args:
            pred_temporal_adj: Predicted adjacency matrix
            gt_temporal_adj: Ground truth adjacency matrix
        
        Returns:
            Dictionary with TRA score and components
        """
        
        # Extract lineages
        pred_lineage = self._extract_lineage_from_adjacency(pred_temporal_adj)
        gt_lineage = self._extract_lineage_from_adjacency(gt_temporal_adj)
        
        # Match tracks
        matches, unmatched_pred, unmatched_gt = self._match_tracks(
            pred_lineage, gt_lineage
        )
        
        # Compute AOGM components
        # ED: number of edges to delete (FP)
        # EA: number of edges to add (FN)  
        # EC: number of edges to alter (semantic errors)
        
        ED = len(unmatched_pred)  # False positive tracks
        EA = len(unmatched_gt)    # False negative tracks
        EC = 0  # Semantic errors in matched tracks
        
        # Count semantic errors (divisions, merges, etc.)
        for pred_id, gt_id in matches.items():
            pred_track = pred_lineage[pred_id]
            gt_track = gt_lineage[gt_id]
            
            # Division errors
            pred_daughters = len(pred_track['daughters'])
            gt_daughters = len(gt_track['daughters'])
            
            if pred_daughters != gt_daughters:
                EC += abs(pred_daughters - gt_daughters)
        
        # Compute AOGM
        AOGM = ED + EA + EC
        
        # AOGM_0 is the cost of constructing GT from scratch
        AOGM_0 = len(gt_lineage)  # Each GT track costs 1 to add
        
        # Compute TRA
        if AOGM_0 > 0:
            TRA = max(0.0, 1.0 - min(1.0, AOGM / AOGM_0))
        else:
            TRA = 0.0
        
        return {
            'TRA': TRA,
            'AOGM': AOGM,
            'AOGM_0': AOGM_0,
            'ED': ED,
            'EA': EA,
            'EC': EC,
            'num_pred_tracks': len(pred_lineage),
            'num_gt_tracks': len(gt_lineage),
            'num_matched_tracks': len(matches)
        }
    
    def compute_division_metrics(
        self,
        pred_temporal_adj: np.ndarray,
        gt_temporal_adj: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 specifically for division events.
        
        Args:
            pred_temporal_adj: Predicted adjacency matrix
            gt_temporal_adj: Ground truth adjacency matrix
        
        Returns:
            Dictionary with division detection metrics
        """
        
        # Extract lineages
        pred_lineage = self._extract_lineage_from_adjacency(pred_temporal_adj)
        gt_lineage = self._extract_lineage_from_adjacency(gt_temporal_adj)
        
        # Count divisions
        pred_divisions = sum(1 for track in pred_lineage.values() 
                           if len(track['daughters']) > 0)
        gt_divisions = sum(1 for track in gt_lineage.values() 
                         if len(track['daughters']) > 0)
        
        # Match tracks to count true positive divisions
        matches, _, _ = self._match_tracks(pred_lineage, gt_lineage)
        
        tp_divisions = 0
        for pred_id, gt_id in matches.items():
            pred_has_division = len(pred_lineage[pred_id]['daughters']) > 0
            gt_has_division = len(gt_lineage[gt_id]['daughters']) > 0
            
            if pred_has_division and gt_has_division:
                tp_divisions += 1
        
        # Compute metrics
        precision = tp_divisions / pred_divisions if pred_divisions > 0 else 0.0
        recall = tp_divisions / gt_divisions if gt_divisions > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) 
              if (precision + recall) > 0 else 0.0)
        
        return {
            'division_precision': precision,
            'division_recall': recall,
            'division_f1': f1,
            'pred_divisions': pred_divisions,
            'gt_divisions': gt_divisions,
            'tp_divisions': tp_divisions
        }
    
    def evaluate_all(
        self,
        pred_temporal_adj: np.ndarray,
        gt_temporal_adj: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Compute all evaluation metrics.
        
        Args:
            pred_temporal_adj: (T-1, N, N, 3) predicted adjacency
            gt_temporal_adj: (T-1, N, N, 3) ground truth adjacency
        
        Returns:
            Dictionary containing all metrics
        """
        
        return {
            'cost_based': self.compute_cost_based_score(
                pred_temporal_adj, gt_temporal_adj
            ),
            'tra': self.compute_tra_score(
                pred_temporal_adj, gt_temporal_adj
            ),
            'divisions': self.compute_division_metrics(
                pred_temporal_adj, gt_temporal_adj
            )
        }


def pretty_print_results(results: Dict[str, Dict]):
    """Pretty print evaluation results."""
    
    print("\n" + "="*70)
    print("TRACKING EVALUATION RESULTS")
    print("="*70)
    
    # Cost-based score
    if 'cost_based' in results:
        print("\n[Cost-Based Scoring]")
        cb = results['cost_based']
        print(f"  Normalized Score: {cb['normalized_score']:.4f}")
        print(f"  Total Score: {cb['total_score']:.2f} / {cb['max_score']:.2f}")
        print(f"\n  Error Breakdown:")
        for error_type, count in cb['error_counts'].items():
            cost = cb['error_costs'][error_type]
            print(f"    {error_type:20s}: {count:3d} errors (cost: {cost:.2f})")
    
    # TRA score
    if 'tra' in results:
        print("\n[TRA Score (Cell Tracking Challenge)]")
        tra = results['tra']
        print(f"  TRA: {tra['TRA']:.4f}")
        print(f"  AOGM: {tra['AOGM']:.2f} / {tra['AOGM_0']:.2f}")
        print(f"  Tracks: {tra['num_matched_tracks']} matched, "
              f"{tra['ED']} FP, {tra['EA']} FN")
        print(f"  Semantic Errors: {tra['EC']}")
    
    # Division metrics
    if 'divisions' in results:
        print("\n[Division Detection]")
        div = results['divisions']
        print(f"  Precision: {div['division_precision']:.4f}")
        print(f"  Recall:    {div['division_recall']:.4f}")
        print(f"  F1 Score:  {div['division_f1']:.4f}")
        print(f"  Divisions: {div['tp_divisions']} TP, "
              f"{div['pred_divisions']} pred, {div['gt_divisions']} GT")
    
    print("\n" + "="*70 + "\n")


# Example usage
if __name__ == "__main__":
    
    print("Testing Tracking Evaluation Metrics")
    print("="*70)
    
    # Create synthetic test data
    T, N = 8, 10
    
    # Ground truth: simple lineage with one division
    gt_adj = np.zeros((T-1, N, N, 3))
    
    # Track 0: frames 0-3, divides at frame 3
    gt_adj[0:3, 0, 0, 0] = 1.0  # same cell
    gt_adj[2, 0, 1, 2] = 1.0    # division to track 1
    gt_adj[2, 0, 2, 2] = 1.0    # division to track 2
    gt_adj[1:3, 0, 0, 1] = 0.0  # mark as not different
    
    # Track 1: frames 3-7 (daughter)
    gt_adj[3:, 1, 1, 0] = 1.0
    
    # Track 2: frames 3-7 (daughter)
    gt_adj[3:, 2, 2, 0] = 1.0
    
    # Track 3: frames 0-7 (independent)
    gt_adj[:, 3, 3, 0] = 1.0
    
    # Mark everything else as "different"
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                if gt_adj[t, i, j, :].sum() == 0:
                    gt_adj[t, i, j, 1] = 1.0
    
    # Predicted: similar but with some errors
    pred_adj = gt_adj.copy()
    
    # Introduce error: miss the division
    pred_adj[2, 0, 1, 2] = 0.0  # miss one daughter
    pred_adj[2, 0, 1, 1] = 1.0  # mark as different instead
    
    # Introduce error: add a false positive track
    pred_adj[:4, 4, 4, 0] = 1.0  # spurious track
    
    # Initialize evaluator
    evaluator = TrackingEvaluator()
    
    # Compute all metrics
    results = evaluator.evaluate_all(pred_adj, gt_adj)
    
    # Print results
    pretty_print_results(results)
    
    print("\nTesting with perfect prediction:")
    perfect_results = evaluator.evaluate_all(gt_adj, gt_adj)
    pretty_print_results(perfect_results)