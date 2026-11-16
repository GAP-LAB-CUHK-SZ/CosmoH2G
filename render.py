#!/usr/bin/env python3
"""Render one release case to a video.

Uses only files under this directory: the case's three sequences, ``meshes/<id>.ply``,
``umi.obj``, and ``MANO_RIGHT.npz`` / ``MANO_LEFT.npz``. The start object (green) and
the end object (orange) stay visible on every frame, together with UMI and the hand.

    /mnt/hongxiang/miniconda3/envs/astribot_simu/bin/python render.py 3d_seq/12_16/56_1_2
    /mnt/hongxiang/miniconda3/envs/astribot_simu/bin/python render.py 3d_seq/1_17/1037_3_4 -o /tmp/out.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
WIDTH = 960
HEIGHT = 640
FPS = 30
R_FIX = np.diag([1.0, -1.0, -1.0])
START_COLOR = np.array([0.42, 0.72, 0.38])
END_COLOR = np.array([0.90, 0.46, 0.20])
UMI_COLOR = np.array([0.0, 159.0 / 255.0, 237.0 / 255.0])
HAND_COLOR = np.array([80.0, 200.0, 120.0]) / 255.0

_PLY_SCALAR = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def inside(path: Path) -> Path:
    """Resolved path that stays inside the release directory."""
    resolved = path.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise SystemExit(f"只能读取 {ROOT} 里的文件：{path}")
    return resolved


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = inside(path)
    blob = path.read_bytes()
    if not blob.startswith(b"ply"):
        raise ValueError(f"{path} 不是 ply")
    marker = b"end_header\n"
    at = blob.find(marker)
    sep = len(marker)
    if at < 0:
        marker = b"end_header\r\n"
        at = blob.find(marker)
        sep = len(marker)
    if at < 0:
        raise ValueError(f"{path} 没有 end_header")
    header = blob[:at].decode("ascii", errors="replace")
    rest = blob[at + sep :]
    endian = "<"
    nvert = nface = None
    vert_props: list[tuple[str, str]] = []
    face_count = "uchar"
    face_index = "int"
    current = None
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("format "):
            fmt = line.split()[1]
            if fmt == "binary_little_endian":
                endian = "<"
            elif fmt == "binary_big_endian":
                endian = ">"
            else:
                raise ValueError(f"{path} 不是二进制 ply")
        elif line.startswith("element "):
            _, name, count = line.split()[:3]
            current = name
            if name == "vertex":
                nvert = int(count)
                vert_props = []
            elif name == "face":
                nface = int(count)
        elif line.startswith("property ") and current == "vertex":
            parts = line.split()
            vert_props.append((parts[2], parts[1]))
        elif line.startswith("property list ") and current == "face":
            parts = line.split()
            face_count, face_index = parts[2], parts[3]
    if nvert is None or nface is None or not vert_props:
        raise ValueError(f"{path} 缺少顶点或面")
    record = np.dtype([(name, endian + _PLY_SCALAR[ptype]) for name, ptype in vert_props])
    need = nvert * record.itemsize
    vert = np.frombuffer(rest[:need], dtype=record, count=nvert)
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=1).astype(np.float64)
    faces: list[tuple[int, int, int]] = []
    off = need
    count_dt = np.dtype(endian + _PLY_SCALAR[face_count])
    index_dt = np.dtype(endian + _PLY_SCALAR[face_index])
    for _ in range(nface):
        count = int(np.frombuffer(rest[off : off + count_dt.itemsize], dtype=count_dt, count=1)[0])
        off += count_dt.itemsize
        idx = np.frombuffer(rest[off : off + index_dt.itemsize * count], dtype=index_dt, count=count)
        off += index_dt.itemsize * count
        idx = [int(v) for v in idx]
        if len(idx) == 3:
            faces.append((idx[0], idx[1], idx[2]))
        elif len(idx) > 3:
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    if not faces:
        raise ValueError(f"{path} 没有三角面")
    return xyz, np.asarray(faces, dtype=np.int32)


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = inside(path)
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            if len(idx) == 3:
                faces.append((idx[0], idx[1], idx[2]))
            elif len(idx) > 3:
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not faces:
        raise ValueError(f"{path} 没有顶点或面")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


class ManoLayer:
    def __init__(self, side: str) -> None:
        path = inside(ROOT / f"MANO_{side.upper()}.npz")
        data = np.load(path)
        self.side = side
        self.v_template = np.asarray(data["v_template"], dtype=np.float64)
        self.posedirs = np.asarray(data["posedirs"], dtype=np.float64)
        self.J_regressor = np.asarray(data["J_regressor"], dtype=np.float64)
        self.weights = np.asarray(data["weights"], dtype=np.float64)
        self.parents = np.asarray(data["parents"], dtype=np.int32)
        self.faces = np.asarray(data["f"], dtype=np.int32)
        self.n_verts = int(self.v_template.shape[0])
        self.n_joints = int(self.J_regressor.shape[0])

    def vertices(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        transl = params[:, :3]
        rot_mats = rot6d_to_matrix(params[:, 3:99].reshape(-1, self.n_joints, 6))
        batch = len(params)
        v_shaped = np.broadcast_to(self.v_template, (batch, self.n_verts, 3))
        joints = np.einsum("bik,ji->bjk", v_shaped, self.J_regressor)
        ident = np.eye(3)
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch, -1)
        v_posed = v_shaped + (pose_feature @ self.posedirs).reshape(batch, self.n_verts, 3)
        transforms = _rigid(rot_mats, joints, self.parents)
        blended = np.einsum("vj,bjkl->bvkl", self.weights, transforms)
        homo = np.concatenate([v_posed, np.ones((batch, self.n_verts, 1))], axis=-1)
        local = np.einsum("bvij,bvj->bvi", blended, homo)[..., :3]
        scale = params[:, 99] if params.shape[1] == 100 else np.ones(batch)
        return scale[:, None, None] * local + transl[:, None, :]


def _rigid(rot_mats: np.ndarray, joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    batch, n_joints = joints.shape[:2]
    rel = joints.copy()
    rel[:, 1:] -= joints[:, parents[1:]]
    mats = np.zeros((batch, n_joints, 4, 4), dtype=np.float64)
    mats[:, :, :3, :3] = rot_mats
    mats[:, :, :3, 3] = rel
    mats[:, :, 3, 3] = 1.0
    chain = [mats[:, 0]]
    for index in range(1, n_joints):
        chain.append(chain[parents[index]] @ mats[:, index])
    transforms = np.stack(chain, axis=1)
    rest = np.einsum("bjik,bjk->bji", transforms[:, :, :3, :3], joints)
    transforms = transforms.copy()
    transforms[:, :, :3, 3] -= rest
    return transforms


def place_object(xyz: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
    scale = float(pose[7]) if len(pose) > 7 else 1.0
    return (scale * xyz) @ rotation.T + pose[:3]


def place_umi(xyz: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
    return (xyz @ R_FIX.T) @ rotation.T + pose[:3]


def mesh_geometry(xyz: np.ndarray, faces: np.ndarray, color: np.ndarray):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.ascontiguousarray(faces, dtype=np.int32))
    colors = np.broadcast_to(np.asarray(color, dtype=np.float64).reshape(1, 3), (len(xyz), 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors))
    mesh.compute_vertex_normals()
    # Normals stay with the original winding. Extra faces only cover the other side.
    doubled = np.vstack([faces, faces[:, ::-1]])
    mesh.triangles = o3d.utility.Vector3iVector(np.ascontiguousarray(doubled, dtype=np.int32))
    return mesh


def view_of(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = 0.5 * (lo + hi)
    extent = float(np.max(hi - lo))
    distance = (extent / 2.0) / np.tan(np.radians(40.0) / 2.0) * 1.15
    azimuth = np.radians(135.0)
    elevation = np.radians(28.0)
    eye = center + distance * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    return center, eye


def resolve_case(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    path = inside(path)
    if not path.is_dir():
        raise SystemExit(f"没有这个 case：{path}")
    return path


def load_case(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, Path]:
    umi_path = inside(case_dir / "umi_pose_seq.npy")
    ends_path = inside(case_dir / "obj_pose_ends.npy")
    hands = [inside(path) for path in case_dir.glob("mano_*.npy")]
    if len(hands) != 1 or not umi_path.is_file() or not ends_path.is_file():
        raise SystemExit(f"{case_dir} 需要 umi_pose_seq.npy、obj_pose_ends.npy 和一份 mano_*.npy")
    side = hands[0].stem.split("_", 1)[1]
    if side not in {"left", "right"}:
        raise SystemExit(f"无法从文件名判断左右手：{hands[0].name}")
    umi = np.load(umi_path)
    mano = np.load(hands[0])
    ends = np.load(ends_path)
    if umi.ndim != 2 or umi.shape[1] != 7 or len(umi) < 2:
        raise SystemExit(f"umi_pose_seq 应为 (N, 7)，实际 {umi.shape}")
    if mano.shape != (len(umi), 100):
        raise SystemExit(f"人手应为 ({len(umi)}, 100)，实际 {mano.shape}")
    if ends.shape != (2, 8):
        raise SystemExit(f"obj_pose_ends 应为 (2, 8)，实际 {ends.shape}")
    object_id = int(case_dir.name.split("_")[0])
    mesh = inside(ROOT / "meshes" / f"{object_id:06d}.ply")
    if not mesh.is_file():
        raise SystemExit(f"缺少物体 mesh：{mesh.name}")
    return umi, mano, ends, side, mesh


def render_case(case_dir: Path, out: Path) -> Path:
    umi_pose, mano, ends, side, mesh_path = load_case(case_dir)
    obj_xyz, obj_faces = read_ply(mesh_path)
    umi_xyz, umi_faces = read_obj(inside(ROOT / "umi.obj"))
    layer = ManoLayer(side)
    hand_xyz = layer.vertices(mano)
    n = len(umi_pose)

    def include(bounds: list[np.ndarray], points: np.ndarray) -> None:
        bounds[0] = np.minimum(bounds[0], points.min(0))
        bounds[1] = np.maximum(bounds[1], points.max(0))

    bounds = [np.full(3, np.inf), np.full(3, -np.inf)]
    obj_lo, obj_hi = obj_xyz.min(0), obj_xyz.max(0)
    obj_corners = np.array(
        [[x, y, z] for x in (obj_lo[0], obj_hi[0]) for y in (obj_lo[1], obj_hi[1]) for z in (obj_lo[2], obj_hi[2])]
    )
    umi_lo, umi_hi = umi_xyz.min(0), umi_xyz.max(0)
    umi_corners = np.array(
        [[x, y, z] for x in (umi_lo[0], umi_hi[0]) for y in (umi_lo[1], umi_hi[1]) for z in (umi_lo[2], umi_hi[2])]
    )
    for pose in ends:
        include(bounds, place_object(obj_corners, pose))
    for pose in umi_pose:
        include(bounds, place_umi(umi_corners, pose))
    include(bounds, hand_xyz.reshape(-1, 3))
    center, eye = view_of(bounds[0], bounds[1])

    import open3d.visualization.rendering as rendering

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.base_roughness = 1.0
    material.base_metallic = 0.0
    material.base_reflectance = 0.12
    umi_material = rendering.MaterialRecord()
    umi_material.shader = "defaultLitTransparency"
    umi_material.base_color = [0.0, 159.0 / 255.0, 237.0 / 255.0, 0.38]
    renderer = rendering.OffscreenRenderer(WIDTH, HEIGHT)
    renderer.scene.set_background([0.94, 0.94, 0.95, 1.0])

    def aim() -> None:
        renderer.scene.camera.look_at(center, eye, [0.0, 0.0, 1.0])
        renderer.scene.camera.set_projection(40.0, WIDTH / HEIGHT, 0.02, 20.0, rendering.Camera.FovType.Vertical)

    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
            "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert encoder.stdin is not None and encoder.stderr is not None
    start_mesh = mesh_geometry(place_object(obj_xyz, ends[0]), obj_faces, START_COLOR)
    end_mesh = mesh_geometry(place_object(obj_xyz, ends[1]), obj_faces, END_COLOR)
    try:
        for frame in range(n):
            renderer.scene.clear_geometry()
            renderer.scene.add_geometry("object_start", start_mesh, material)
            renderer.scene.add_geometry("object_end", end_mesh, material)
            renderer.scene.add_geometry(
                "umi", mesh_geometry(place_umi(umi_xyz, umi_pose[frame]), umi_faces, UMI_COLOR), umi_material
            )
            renderer.scene.add_geometry(
                "hand", mesh_geometry(hand_xyz[frame], layer.faces, HAND_COLOR), material
            )
            aim()
            image = np.asarray(renderer.render_to_image())
            encoder.stdin.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
            if frame % 50 == 0 or frame == n - 1:
                print(f"[render] {frame + 1}/{n}", flush=True)
    finally:
        encoder.stdin.close()
        err = encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg 写入失败：{err[-400:]}")
    print(f"[render] {out}  {n} 帧", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="用 release 目录里的文件渲染一条 case")
    parser.add_argument("case", help="例如 3d_seq/12_16/56_1_2")
    parser.add_argument("-o", "--out", type=Path, default=None, help="输出 mp4，默认写到当前目录")
    args = parser.parse_args()
    case_dir = resolve_case(args.case)
    out = args.out
    if out is None:
        out = Path.cwd() / f"{case_dir.parent.name}_{case_dir.name}.mp4"
    elif not out.is_absolute():
        out = Path.cwd() / out
    try:
        render_case(case_dir, out)
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
