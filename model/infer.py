"""Predict a UMI gripper trajectory for CosmoH2G episodes.

Usage:
    python infer.py --data_root /path/to/CosmoH2G --cases 1_16/977_9_10 --vis
    python infer.py --data_root /path/to/CosmoH2G --case_list splits/test.txt

For each case, writes to <out_dir>/<date>_<case>/:
    umi_pose_pred.npy   (T, 7) world pose of umi_base, `x y z qw qx qy qz`, same convention as umi_pose_seq.npy
    keypoints_pred.npy  (T, 3, 3) world center / left / right gripper keypoints
    frame_idx.npy       (T,) source frame of each predicted step
    scene.ply           start/end object, first/last hand and predicted grippers (with --vis)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from cosmoh2g import CosmoH2G
from cosmoh2g.data import load_case, read_case_list, read_mesh
from cosmoh2g.geometry import place_object, place_umi_from_keypoints, umi_pose_from_keypoints

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Unpacked CosmoH2G release root")
    parser.add_argument("--cases", nargs="*", default=[], help="Episodes such as 1_16/977_9_10")
    parser.add_argument("--case_list", type=Path, default=None, help="Text file with one episode per line")
    parser.add_argument("--ckpt_dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "--refine_endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Contact / penetration refinement of the first and last grasp",
    )
    return parser.parse_args()


def paint(verts, faces, color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.ascontiguousarray(verts, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.ascontiguousarray(faces, dtype=np.int32))
    mesh.paint_uniform_color(list(color))
    mesh.compute_vertex_normals()
    return mesh


def write_scene(path, item, keypoints, umi_verts, umi_faces):
    ends = item["obj_pose_ends"]
    scene = paint(place_object(item["obj_verts"], ends[0]), item["obj_faces"], (0.42, 0.72, 0.38))
    scene += paint(place_object(item["obj_verts"], ends[1]), item["obj_faces"], (0.86, 0.78, 0.28))
    center = item["obj_center"]
    scene += paint(item["hand_verts"][0] + center, item["hand_faces"], (0.96, 0.72, 0.45))
    scene += paint(item["hand_verts"][-1] + center, item["hand_faces"], (0.86, 0.48, 0.32))
    for t, kpts in enumerate(keypoints):
        alpha = t / max(len(keypoints) - 1, 1)
        color = (1.0 - alpha) * np.array([0.45, 0.82, 0.95]) + alpha * np.array([0.05, 0.22, 0.82])
        scene += paint(place_umi_from_keypoints(umi_verts, kpts), umi_faces, color)
    o3d.io.write_triangle_mesh(str(path), scene, write_vertex_colors=True)


def main():
    args = parse_args()
    cases = list(args.cases)
    if args.case_list is not None:
        cases += read_case_list(args.case_list)
    if not cases:
        raise SystemExit("Pass --cases or --case_list")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    o3d.utility.random.seed(args.seed)
    umi_verts, umi_faces = read_mesh(os.path.join(args.data_root, "umi.obj"))
    model = CosmoH2G(args.ckpt_dir, umi_verts=umi_verts)

    for start in range(0, len(cases), args.batch_size):
        items = [load_case(args.data_root, case) for case in cases[start:start + args.batch_size]]
        for item, pred in zip(items, model.predict(items, refine=args.refine_endpoints)):
            out_dir = args.out_dir / item["case"].replace("/", "_")
            out_dir.mkdir(parents=True, exist_ok=True)
            keypoints = pred["keypoints"] + item["obj_center"]
            np.save(out_dir / "keypoints_pred.npy", keypoints.astype(np.float32))
            np.save(out_dir / "umi_pose_pred.npy", np.stack([umi_pose_from_keypoints(k) for k in keypoints]))
            np.save(out_dir / "frame_idx.npy", item["frame_idx"])
            if args.vis:
                write_scene(out_dir / "scene.ply", item, keypoints, umi_verts, umi_faces)
            print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
