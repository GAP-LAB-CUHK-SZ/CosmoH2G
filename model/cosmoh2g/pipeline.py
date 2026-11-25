import os

import numpy as np
import torch

from cosmoh2g.diffusion import Sampler
from cosmoh2g.geometry import (
    hand_contact_centers,
    init_centers,
    keypoints_from_dirs,
    place_object,
    smooth_centers,
    transfer_keypoints,
    unit_dirs,
)
from cosmoh2g.network import CosmoH2GDenoiser
from cosmoh2g.refine import refine_endpoints


def load_stage(stage, path, device):
    model = CosmoH2GDenoiser(stage)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return model.to(device).eval()


class CosmoH2G:
    """Stage I keyframe on the start object, rigid transfer to the end object, then Stage II sequence."""

    def __init__(self, ckpt_dir, device="cuda", umi_verts=None):
        """`umi_verts`: raw `umi.obj` vertices, required by `predict(..., refine=True)`."""
        self.device = torch.device(device)
        self.stage1 = load_stage(1, os.path.join(ckpt_dir, "stage1.pt"), self.device)
        self.stage2 = load_stage(2, os.path.join(ckpt_dir, "stage2.pt"), self.device)
        self.sampler = Sampler()
        self.umi_verts = umi_verts

    @torch.no_grad()
    def predict(self, items, refine=True, progress=False):
        """Run on a list of `data.load_case` outputs.

        Returns one dict per item with `start` (3, 3), `end` (3, 3) and `keypoints` (T, 3, 3):
        center / left / right gripper keypoints in the start-object centroid frame.
        With `refine`, the first and last grasps are translated for contact alignment and
        to resolve penetration before the middle frames are placed.
        """
        if refine and self.umi_verts is None:
            raise ValueError("refine=True needs umi_verts")
        batch = len(items)
        obj = torch.from_numpy(np.stack([item["obj_pts"] for item in items])).to(self.device)
        hand = torch.from_numpy(np.stack([item["hand_seq"] for item in items])).to(self.device)
        cont = torch.from_numpy(np.stack([item["hoi_cont"] for item in items])).to(self.device)[..., None]
        num_frames = hand.shape[1]

        start = self.sampler.sample(
            self.stage1,
            (batch, 1, 9),
            self.device,
            progress=progress,
            c_obj_pts=obj,
            c_hoi_cont=cont,
            c_hand_seq=hand[:, :1],
        )
        start_kpts = start.reshape(batch, 3, 3).cpu().numpy()
        end_kpts = np.zeros_like(start_kpts)
        for i, item in enumerate(items):
            ends = item["obj_pose_ends"]
            obj_center = place_object(item["obj_verts"], ends[0]).mean(axis=0)
            end_kpts[i] = transfer_keypoints(start_kpts[i], ends[0], ends[1], obj_center)

        start_dirs = unit_dirs(start_kpts).astype(np.float32)
        end_dirs = unit_dirs(end_kpts).astype(np.float32)
        sample = self.sampler.sample(
            self.stage2,
            (batch, num_frames, 6),
            self.device,
            progress=progress,
            c_obj_pts=obj,
            c_hoi_cont=cont,
            c_hand_seq=hand,
            c_gripper_start=torch.from_numpy(start_dirs).to(self.device),
            c_gripper_end=torch.from_numpy(end_dirs).to(self.device),
        )
        dirs = sample.reshape(batch, num_frames, 2, 3).cpu().numpy()
        dirs /= np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)
        dirs[:, 0] = start_dirs
        dirs[:, -1] = end_dirs

        if refine:
            for i, item in enumerate(items):
                start_kpts[i], end_kpts[i] = refine_endpoints(
                    item, start_kpts[i], end_kpts[i], self.umi_verts, self.device
                )

        results = []
        for i, item in enumerate(items):
            centers = init_centers(dirs[i], hand_contact_centers(item["hand_seq"], item["obj_pts"]))
            centers[0] = start_kpts[i, 0]
            centers[-1] = end_kpts[i, 0]
            centers = smooth_centers(centers)
            frames = [start_kpts[i]]
            frames += [keypoints_from_dirs(centers[t], dirs[i, t]) for t in range(1, num_frames - 1)]
            frames.append(end_kpts[i])
            results.append({"start": start_kpts[i], "end": end_kpts[i], "keypoints": np.stack(frames)})
        return results
