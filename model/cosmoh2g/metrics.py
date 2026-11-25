import numpy as np

from cosmoh2g.geometry import R_FIX, UMI_KEYPOINTS, rigid_align

TAU = 2.30


def approach_axis(kpts):
    """Unit normal of the (center, left, right) plane."""
    center, left, right = np.asarray(kpts, dtype=np.float64)
    axis = np.cross(0.5 * (left + right) - center, left - right)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        rot, _ = rigid_align(UMI_KEYPOINTS @ R_FIX.T, kpts)
        axis = (np.array([0.0, 1.0, 0.0]) @ R_FIX.T) @ rot.T
        norm = np.linalg.norm(axis)
    return axis / max(norm, 1e-8)


def axis_angle_deg(kpts_a, kpts_b):
    cos = float(np.clip(np.dot(approach_axis(kpts_a), approach_axis(kpts_b)), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos)))
    return min(angle, 180.0 - angle)


def grasp_orientation_accuracy(pred_start, gt_start):
    """GOA (deg): orientation error of the initial grasp."""
    return axis_angle_deg(pred_start, gt_start)


def terminal_orientation_accuracy(pred_end, gt_end):
    """TOPA (deg): orientation error of the terminal grasp."""
    return axis_angle_deg(pred_end, gt_end)


def dtw_distance(seq_a, seq_b):
    n, m = len(seq_a), len(seq_b)
    prev = np.full(m + 1, 1e9)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, 1e9)
        for j in range(1, m + 1):
            step = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
            cur[j] = step + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m])


def trajectory_similarity(pred_kpts, gt_kpts):
    """TS: DTW similarity of the fingertip-midpoint trajectories."""
    pred = 0.5 * (np.asarray(pred_kpts)[:, 1] + np.asarray(pred_kpts)[:, 2])
    gt = 0.5 * (np.asarray(gt_kpts)[:, 1] + np.asarray(gt_kpts)[:, 2])
    distance = dtw_distance(pred, gt) / len(pred)
    return 1.0 - min(distance / TAU, 1.0)
