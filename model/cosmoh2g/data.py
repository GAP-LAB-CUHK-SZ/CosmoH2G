"""Load one CosmoH2G episode into the network inputs.

Expected layout under `data_root` (the ModelScope release, unpacked):
    3d_seq/<date>/<case>/{umi_pose_seq.npy, mano_right.npy|mano_left.npy, obj_pose_ends.npy}
    meshes/<object_id:06d>.ply
    umi.obj, MANO_RIGHT.npz, MANO_LEFT.npz

Everything is expressed in the start-object centroid frame.
"""
import os
from functools import lru_cache

import numpy as np
import open3d as o3d

from cosmoh2g.geometry import UMI_KEYPOINTS, contact_map, place_object, place_umi

NUM_POINTS = 2048
NUM_FRAMES = 30


def _rot6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


class ManoLayer:
    """MANO forward kinematics for the release `MANO_{SIDE}.npz` files (betas = 0)."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.v_template = np.asarray(data["v_template"], dtype=np.float64)
        self.posedirs = np.asarray(data["posedirs"], dtype=np.float64)
        self.J_regressor = np.asarray(data["J_regressor"], dtype=np.float64)
        self.weights = np.asarray(data["weights"], dtype=np.float64)
        self.parents = np.asarray(data["parents"], dtype=np.int32)
        self.faces = np.asarray(data["f"], dtype=np.int32)
        self.n_verts = int(self.v_template.shape[0])
        self.n_joints = int(self.J_regressor.shape[0])

    def _rigid(self, rot_mats, joints):
        batch, n_joints = joints.shape[:2]
        rel = joints.copy()
        rel[:, 1:] -= joints[:, self.parents[1:]]
        mats = np.zeros((batch, n_joints, 4, 4), dtype=np.float64)
        mats[:, :, :3, :3] = rot_mats
        mats[:, :, :3, 3] = rel
        mats[:, :, 3, 3] = 1.0
        chain = [mats[:, 0]]
        for index in range(1, n_joints):
            chain.append(chain[self.parents[index]] @ mats[:, index])
        transforms = np.stack(chain, axis=1)
        rest = np.einsum("bjik,bjk->bji", transforms[:, :, :3, :3], joints)
        transforms = transforms.copy()
        transforms[:, :, :3, 3] -= rest
        return transforms

    def vertices(self, params):
        params = np.asarray(params, dtype=np.float64)
        if params.ndim == 1:
            params = params[None]
        batch = len(params)
        rot_mats = _rot6d_to_matrix(params[:, 3:99].reshape(-1, self.n_joints, 6))
        v_shaped = np.broadcast_to(self.v_template, (batch, self.n_verts, 3))
        joints = np.einsum("bik,ji->bjk", v_shaped, self.J_regressor)
        pose_feature = (rot_mats[:, 1:] - np.eye(3)).reshape(batch, -1)
        v_posed = v_shaped + (pose_feature @ self.posedirs).reshape(batch, self.n_verts, 3)
        blended = np.einsum("vj,bjkl->bvkl", self.weights, self._rigid(rot_mats, joints))
        homo = np.concatenate([v_posed, np.ones((batch, self.n_verts, 1))], axis=-1)
        local = np.einsum("bvij,bvj->bvi", blended, homo)[..., :3]
        scale = params[:, 99] if params.shape[1] == 100 else np.ones(batch)
        return scale[:, None, None] * local + params[:, None, :3]


@lru_cache(maxsize=2)
def _mano(data_root, side):
    return ManoLayer(os.path.join(data_root, f"MANO_{side.upper()}.npz"))


def read_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.triangles, dtype=np.int32).copy()
    if verts.size == 0 or faces.size == 0:
        raise ValueError(f"Failed to read mesh: {path}")
    return verts, faces


def sample_points(verts, faces, num_points=NUM_POINTS):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.ascontiguousarray(verts, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.ascontiguousarray(faces, dtype=np.int32))
    return np.asarray(mesh.sample_points_uniformly(number_of_points=num_points).points, dtype=np.float32)


def frame_indices(n_src, n_dst=NUM_FRAMES):
    if n_src <= 1:
        return np.zeros(n_dst, dtype=int)
    return np.linspace(0, n_src - 1, n_dst, dtype=int)


def read_case_list(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def load_case(data_root, case):
    """`case` is `<date>/<object_id>_<slot_start>_<slot_end>`, e.g. `1_16/977_9_10`."""
    folder = os.path.join(data_root, "3d_seq", case)
    hands = [name for name in os.listdir(folder) if name.startswith("mano_") and name.endswith(".npy")]
    if len(hands) != 1:
        raise FileNotFoundError(f"Expected one mano_*.npy in {folder}")
    side = hands[0][len("mano_"):-len(".npy")]
    mano = np.load(os.path.join(folder, hands[0]))
    ends = np.load(os.path.join(folder, "obj_pose_ends.npy"))
    umi_path = os.path.join(folder, "umi_pose_seq.npy")
    object_id = int(os.path.basename(case).split("_")[0])

    obj_verts, obj_faces = read_mesh(os.path.join(data_root, "meshes", f"{object_id:06d}.ply"))
    start_verts = place_object(obj_verts, ends[0])
    obj_center = start_verts.mean(axis=0)
    obj_pts = sample_points(start_verts - obj_center, obj_faces)

    frame_idx = frame_indices(len(mano))
    layer = _mano(data_root, side)
    hand_verts = layer.vertices(mano[frame_idx]) - obj_center
    hand_seq = np.stack([sample_points(hand_verts[t], layer.faces) for t in range(len(frame_idx))])

    item = {
        "case": case,
        "side": side,
        "obj_center": obj_center,
        "obj_pose_ends": ends,
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
        "obj_pts": obj_pts,
        "hand_seq": hand_seq,
        "hand_verts": hand_verts,
        "hand_faces": layer.faces,
        "hoi_cont": contact_map(hand_seq[0], obj_pts),
        "frame_idx": frame_idx,
    }
    if os.path.isfile(umi_path):
        umi_pose = np.load(umi_path)
        item["gt_keypoints"] = np.stack(
            [place_umi(UMI_KEYPOINTS, umi_pose[int(t)]) - obj_center for t in frame_idx]
        ).astype(np.float32)
    return item
