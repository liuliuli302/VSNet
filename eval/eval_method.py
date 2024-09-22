from typing import Tuple, Iterable, List
from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver
import numpy as np


def knapsack(values: Iterable[int],
             weights: Iterable[int],
             capacity: int
             ) -> List[int]:
    knapsack_solver = KnapsackSolver(
        KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 'test'
    )

    values = list(values)
    weights = list(weights)
    capacity = int(capacity)

    knapsack_solver.Init(values, [weights], [capacity])
    knapsack_solver.Solve()
    packed_items = [x for x in range(0, len(weights))
                    if knapsack_solver.BestSolutionContains(x)]

    return packed_items


def iou_lr(anchor_bbox: np.ndarray, target_bbox: np.ndarray) -> np.ndarray:
    anchor_left, anchor_right = anchor_bbox[:, 0], anchor_bbox[:, 1]
    target_left, target_right = target_bbox[:, 0], target_bbox[:, 1]

    inter_left = np.maximum(anchor_left, target_left)
    inter_right = np.minimum(anchor_right, target_right)
    union_left = np.minimum(anchor_left, target_left)
    union_right = np.maximum(anchor_right, target_right)

    intersect = inter_right - inter_left
    intersect[intersect < 0] = 0
    union = union_right - union_left
    union[union <= 0] = 1e-6

    iou = intersect / union
    return iou


def nms(scores: np.ndarray,
        bboxes: np.ndarray,
        thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    valid_idx = bboxes[:, 0] < bboxes[:, 1]
    scores = scores[valid_idx]
    bboxes = bboxes[valid_idx]

    arg_desc = scores.argsort()[::-1]

    scores_remain = scores[arg_desc]
    bboxes_remain = bboxes[arg_desc]

    keep_bboxes = []
    keep_scores = []

    while bboxes_remain.size > 0:
        bbox = bboxes_remain[0]
        score = scores_remain[0]
        keep_bboxes.append(bbox)
        keep_scores.append(score)

        iou = iou_lr(bboxes_remain, np.expand_dims(bbox, axis=0))

        keep_indices = (iou < thresh)
        bboxes_remain = bboxes_remain[keep_indices]
        scores_remain = scores_remain[keep_indices]

    keep_bboxes = np.asarray(keep_bboxes, dtype=bboxes.dtype)
    keep_scores = np.asarray(keep_scores, dtype=scores.dtype)

    return keep_scores, keep_bboxes


def offset2bbox(offsets: np.ndarray) -> np.ndarray:
    offset_left, offset_right = offsets[:, :, 0], offsets[:, :, 1]  # [B, N]
    B, seq_len, _ = offsets.shape
    indices = np.arange(seq_len).reshape(1, seq_len)
    indices = np.repeat(indices, B, axis=0)  # [B, N]
    bbox_left = indices - offset_left  # [B, N]
    bbox_right = indices + offset_right + 1  # [B, N]
    bboxes = np.stack((bbox_left, bbox_right), axis=-1)  # [B, N, 2]

    return bboxes


def bbox2summary(seq_len: int,
                 pred_cls: np.ndarray,
                 pred_bboxes: np.ndarray,
                 change_points: np.ndarray,
                 n_frames: int,
                 nfps: np.ndarray,
                 picks: np.ndarray,
                 proportion: float = 0.15,
                 seg_score_mode: str = 'mean',
                 ) -> np.ndarray:
    pred_score = np.zeros(seq_len, dtype=np.float32)
    for bbox_idx in range(len(pred_bboxes)):
        lo, hi = pred_bboxes[bbox_idx, 0], pred_bboxes[bbox_idx, 1]
        pred_score[lo:hi] = np.maximum(pred_score[lo:hi], [pred_cls[bbox_idx]])

    pred_summ_upsampled, pred_score_upsampled = get_keyshot_summ(pred_score, change_points, n_frames, nfps, picks,
                                                                 proportion=proportion, seg_score_mode=seg_score_mode)
    pred_summ = pred_summ_upsampled[::15]
    return pred_summ, pred_summ_upsampled, pred_score, pred_score_upsampled


def get_keyshot_summ(pred: np.ndarray,
                     cps: np.ndarray,
                     n_frames: int,
                     nfps: np.ndarray,
                     picks: np.ndarray,
                     proportion: float = 0.15,
                     seg_score_mode: str = 'mean',
                     ) -> np.ndarray:
    picks = np.asarray(picks, dtype=np.int32)
    assert pred.shape == picks.shape, "pred:{} picks:{}".format(pred.shape, picks.shape)

    # Get original frame scores from downsampled sequence
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(len(picks)):
        pos_lo = picks[i]
        pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        frame_scores[pos_lo:pos_hi] = pred[i]

    # Assign scores to video shots as the average of the frames.
    seg_scores = np.zeros(len(cps), dtype=np.int32)
    for seg_idx, (first, last) in enumerate(cps):
        scores = frame_scores[first:last + 1]
        if seg_score_mode == 'mean':
            seg_scores[seg_idx] = int(1000 * scores.mean())
        elif seg_score_mode == 'sum':
            seg_scores[seg_idx] = int(1000 * scores.sum())

    # Apply knapsack algorithm to find the best shots
    limits = int(round(n_frames * proportion))
    packed = knapsack(seg_scores, nfps, limits)
    # pdb.set_trace()
    # sum(nfps[packed])

    # Get key-shot based summary
    summary = np.zeros(n_frames, dtype=bool)
    for seg_idx in packed:
        first, last = cps[seg_idx]
        summary[first:last + 1] = True

    # pdb.set_trace()
    return summary, frame_scores


def f1_score(pred: np.ndarray, test: np.ndarray) -> float:
    assert pred.shape == test.shape
    pred = np.asarray(pred, dtype=bool)
    test = np.asarray(test, dtype=bool)
    overlap = (pred & test).sum()
    if overlap == 0:
        return 0.0
    precision = overlap / pred.sum()
    recall = overlap / test.sum()
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def get_summ_f1score(pred_summ: np.ndarray,
                     test_summ: np.ndarray,
                     eval_metric: str = 'avg'
                     ) -> float:
    pred_summ = np.asarray(pred_summ, dtype=bool)
    test_summ = np.asarray(test_summ, dtype=bool)
    _, n_frames = test_summ.shape

    if pred_summ.size > n_frames:
        pred_summ = pred_summ[:n_frames]
    elif pred_summ.size < n_frames:
        pred_summ = np.pad(pred_summ, (0, n_frames - pred_summ.size))

    f1s = [f1_score(user_summ, pred_summ) for user_summ in test_summ]

    if eval_metric == 'avg':
        final_f1 = np.mean(f1s)
    elif eval_metric == 'max':
        final_f1 = np.max(f1s)
    else:
        raise ValueError(f'Invalid eval metric {eval_metric}')

    return float(final_f1)
