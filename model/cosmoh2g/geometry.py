import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

R_FIX = np.diag([1.0, -1.0, -1.0])

# center / left fingertip / right fingertip on the raw umi.obj template
UMI_KEYPOINTS = np.array(
    [
        [0.00061255, 0.00232993, 0.02475288],
        [-0.03886113, 0.00232993, -0.17261554],
        [0.04008624, 0.00232993, -0.17261554],
    ],
    dtype=np.float64,
)
FINGER_LENGTHS = (
    float(np.linalg.norm(UMI_KEYPOINTS[1] - UMI_KEYPOINTS[0])),
    float(np.linalg.norm(UMI_KEYPOINTS[2] - UMI_KEYPOINTS[0])),
)

CONTACT_DIST = 0.04
SMOOTH_STEPS = 200
SMOOTH_LR = 1e-2
LAMBDA_SMOOTH = 1.0
LAMBDA_REF = 10.0


def pose_parts(pose):
    pose = np.asarray(pose, dtype=np.float64)
    rotation = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
    scale = float(pose[7]) if len(pose) > 7 else 1.0
    return rotation, scale, pose[:3]


def place_object(xyz, pose):
    """Object pose `x y z qw qx qy qz scale` applied to template vertices."""
    rotation, scale, translation = pose_parts(pose)
    return (scale * np.asarray(xyz, dtype=np.float64)) @ rotation.T + translation


def place_umi(xyz, pose):
    """UMI pose `x y z qw qx qy qz` applied to template vertices or keypoints."""
    rotation, _, translation = pose_parts(pose)
    return (np.asarray(xyz, dtype=np.float64) @ R_FIX.T) @ rotation.T + translation


def rigid_align(src, dst):
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - src_mean).T @ (dst - dst_mean))
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        rot = vt.T @ u.T
    return rot, dst_mean - src_mean @ rot.T


def umi_pose_from_keypoints(kpts):
    """Inverse of `place_umi(UMI_KEYPOINTS, pose)`: returns `x y z qw qx qy qz`."""
    rot, trans = rigid_align(UMI_KEYPOINTS @ R_FIX.T, kpts)
    qx, qy, qz, qw = Rotation.from_matrix(rot).as_quat()
    return np.array([trans[0], trans[1], trans[2], qw, qx, qy, qz], dtype=np.float64)


def place_umi_from_keypoints(template_verts, kpts):
    rot, trans = rigid_align(UMI_KEYPOINTS @ R_FIX.T, kpts)
    return (np.asarray(template_verts, dtype=np.float64) @ R_FIX.T) @ rot.T + trans


def transfer_keypoints(kpts, start_pose, end_pose, obj_center):
    """Carry a grasp with the object from its start pose to its end pose."""
    rotation_s, scale_s, translation_s = pose_parts(start_pose)
    rotation_e, scale_e, translation_e = pose_parts(end_pose)
    world = np.asarray(kpts, dtype=np.float64) + obj_center
    local = ((world - translation_s) @ rotation_s) / scale_s
    return (scale_e * local) @ rotation_e.T + translation_e - obj_center


def unit_dirs(kpts):
    kpts = np.asarray(kpts, dtype=np.float64)
    rel = kpts[..., 1:, :] - kpts[..., :1, :]
    return rel / np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-8)


def keypoints_from_dirs(center, dirs):
    center = np.asarray(center, dtype=np.float64).reshape(3)
    dirs = np.asarray(dirs, dtype=np.float64).reshape(2, 3)
    return np.stack([center, center + dirs[0] * FINGER_LENGTHS[0], center + dirs[1] * FINGER_LENGTHS[1]])


def contact_map(hand_pts, obj_pts, max_dist=CONTACT_DIST):
    dists, _ = cKDTree(np.asarray(hand_pts, dtype=np.float64)).query(np.asarray(obj_pts, dtype=np.float64), k=1)
    return np.clip(1.0 - dists / max_dist, 0.0, 1.0).astype(np.float32)


def hand_contact_centers(hand_seq, obj_pts):
    """Per-frame centroid of hand points near the object; the hand centroid once it lets go."""
    tree = cKDTree(np.asarray(obj_pts, dtype=np.float64))
    centers = np.zeros((len(hand_seq), 3), dtype=np.float64)
    for t, hand in enumerate(hand_seq):
        dist, _ = tree.query(hand, k=1)
        mask = dist < CONTACT_DIST
        centers[t] = hand[mask].mean(axis=0) if int(mask.sum()) >= 16 else hand.mean(axis=0)
    return centers


def init_centers(dirs, hand_cc):
    """Gripper centers that put the fingertip midpoint on the hand contact center."""
    offset = 0.5 * (FINGER_LENGTHS[0] * dirs[:, 0] + FINGER_LENGTHS[1] * dirs[:, 1])
    return hand_cc - offset


def _min_sqdist_to_polyline(points, knots):
    a, b = knots[:-1], knots[1:]
    ab = b - a
    ap = points[:, None, :] - a[None, :, :]
    scale = ((ap * ab[None]).sum(dim=-1) / ab.pow(2).sum(dim=-1).clamp_min(1e-8)).clamp(0.0, 1.0)
    proj = a[None, :, :] + scale[..., None] * ab[None, :, :]
    return (points[:, None, :] - proj).pow(2).sum(dim=-1).min(dim=1).values


def smooth_centers(centers):
    """Smooth the middle gripper centers with both endpoints held fixed."""
    if len(centers) < 3:
        return centers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.enable_grad():
        knots = torch.tensor(centers, dtype=torch.float32, device=device)
        start, end = knots[:1].detach(), knots[-1:].detach()
        mid = knots[1:-1].detach().clone().requires_grad_(True)
        ref = knots.detach()
        opt = torch.optim.Adam([mid], lr=SMOOTH_LR)
        for _ in range(SMOOTH_STEPS):
            opt.zero_grad()
            seq = torch.cat([start, mid, end], dim=0)
            acc = seq[:-2] - 2.0 * seq[1:-1] + seq[2:]
            loss = LAMBDA_SMOOTH * acc.pow(2).sum() + LAMBDA_REF * _min_sqdist_to_polyline(mid, ref).sum()
            loss.backward()
            opt.step()
    out = np.array(centers, copy=True)
    out[1:-1] = mid.detach().cpu().numpy()
    return out
