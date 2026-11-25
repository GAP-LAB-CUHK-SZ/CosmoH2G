import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

from cosmoh2g.geometry import CONTACT_DIST, place_object, place_umi_from_keypoints

CONTACT_THRESH = 0.5
NUM_GRIPPER_POINTS = 512
NUM_SURFACE_POINTS = 4096
ENDPOINT_STEPS = 200
ENDPOINT_LR = 5e-3
LAMBDA_PEN = 10.0
LAMBDA_REG = 0.05


def outward_normals(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.ascontiguousarray(verts, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.ascontiguousarray(faces, dtype=np.int32))
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if np.mean(np.sum((verts - verts.mean(axis=0)) * normals, axis=1)) < 0:
        normals = -normals
    return normals


def subsample(rng, limit, *arrays):
    if len(arrays[0]) <= limit:
        return arrays
    choice = rng.choice(len(arrays[0]), limit, replace=False)
    return tuple(array[choice] for array in arrays)


def start_contact(obj_pts, hoi_cont, hand_pts):
    mask = np.asarray(hoi_cont).reshape(-1) >= CONTACT_THRESH
    if int(mask.sum()) >= 8:
        return obj_pts[mask]
    dist, _ = cKDTree(hand_pts).query(obj_pts, k=1)
    return obj_pts[np.argsort(dist)[:64]]


def end_contact(end_verts, hand_pts):
    dist, _ = cKDTree(hand_pts).query(end_verts, k=1)
    near = dist < CONTACT_DIST
    if int(near.sum()) >= 8:
        return end_verts[near]
    return end_verts[np.argsort(dist)[:256]]


def refine_translation(kpts, contact, surface, normals, gripper_local, device):
    """Shift one grasp so both fingers line up with the contact region along the closing axis
    while the gripper stays outside the object. Orientation is unchanged."""
    kpts_t = torch.tensor(kpts, dtype=torch.float32, device=device)
    closing = kpts_t[2] - kpts_t[1]
    axis = closing / closing.norm().clamp_min(1e-8)
    contact = torch.tensor(contact, dtype=torch.float32, device=device)
    surface = torch.tensor(surface, dtype=torch.float32, device=device)
    normals = torch.tensor(normals, dtype=torch.float32, device=device)
    gripper = torch.tensor(place_umi_from_keypoints(gripper_local, kpts), dtype=torch.float32, device=device)
    delta = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=ENDPOINT_LR)
    for _ in range(ENDPOINT_STEPS):
        opt.zero_grad()
        contact_loss = sum(((kpts_t[k] + delta - contact) @ axis).abs().min() for k in (1, 2))
        points = gripper + delta
        nearest = torch.cdist(points, surface).argmin(dim=1)
        signed = ((points - surface[nearest]) * normals[nearest]).sum(dim=-1)
        loss = contact_loss + LAMBDA_PEN * torch.relu(-signed).pow(2).mean() + LAMBDA_REG * delta.pow(2).mean()
        loss.backward()
        opt.step()
    return delta.detach().cpu().numpy().astype(np.float64)


def refine_endpoints(item, start_kpts, end_kpts, umi_verts, device):
    """Contact alignment and penetration refinement of the first and last grasp."""
    rng = np.random.default_rng(0)
    (gripper_local,) = subsample(rng, NUM_GRIPPER_POINTS, umi_verts)
    ends = item["obj_pose_ends"]
    center = item["obj_center"]
    faces = item["obj_faces"]
    start_verts = place_object(item["obj_verts"], ends[0]) - center
    end_verts = place_object(item["obj_verts"], ends[1]) - center
    start_surf, start_nrm = subsample(rng, NUM_SURFACE_POINTS, start_verts, outward_normals(start_verts, faces))
    end_surf, end_nrm = subsample(rng, NUM_SURFACE_POINTS, end_verts, outward_normals(end_verts, faces))

    with torch.enable_grad():
        start_shift = refine_translation(
            start_kpts,
            start_contact(item["obj_pts"], item["hoi_cont"], item["hand_seq"][0]),
            start_surf,
            start_nrm,
            gripper_local,
            device,
        )
        end_shift = refine_translation(
            end_kpts,
            end_contact(end_verts, item["hand_seq"][-1]),
            end_surf,
            end_nrm,
            gripper_local,
            device,
        )
    return start_kpts + start_shift, end_kpts + end_shift
