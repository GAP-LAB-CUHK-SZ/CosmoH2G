# CosmoH2G Baseline: Inference and Evaluation

Pretrained two-stage baseline for hand-to-gripper transfer on the CosmoH2G dataset.

- **Stage I** predicts the initial gripper keyframe (center / left / right keypoints) from the start object, the hand–object contact map and the first hand frame. The keyframe is carried to the terminal object pose by the object's own start-to-end rigid transform.
- **Stage II** predicts the full sequence of finger directions conditioned on the object, the contact map, the hand sequence and the two keyframes. Gripper positions follow the hand contact region and are smoothed with both keyframes fixed.

## Installation

Tested with Python 3.10, PyTorch 2.6 and CUDA 12.4. A CUDA GPU is required.

```bash
conda create -n cosmoh2g python=3.10 -y
conda activate cosmoh2g
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install git+https://github.com/Silverster98/pointops@86c68b1e63092152b4ab14ff09b832c5d2d0dc97
```

## Data and Checkpoints

Download and unpack the dataset from [ModelScope `GAP-Lab/CosmoH2G`](https://www.modelscope.cn/datasets/GAP-Lab/CosmoH2G) as described in the [top-level README](../README.md). `--data_root` points at the directory that holds `3d_seq/`, `meshes/`, `umi.obj`, `MANO_RIGHT.npz` and `MANO_LEFT.npz`.

The two checkpoints are in `checkpoints/`:

```text
checkpoints/
├── stage1.pt
└── stage2.pt
```

## Inference

```bash
python infer.py --data_root /path/to/CosmoH2G --cases 1_16/977_9_10 --vis
python infer.py --data_root /path/to/CosmoH2G --case_list splits/test.txt
```

Each episode is resampled to 30 steps. Outputs go to `outputs/<date>_<case>/`:

| File | Shape | Content |
|------|-------|---------|
| `umi_pose_pred.npy` | `(30, 7)` | Predicted world pose of `umi_base`, `x y z qw qx qy qz`, same convention as `umi_pose_seq.npy` |
| `keypoints_pred.npy` | `(30, 3, 3)` | Predicted world center / left / right gripper keypoints |
| `frame_idx.npy` | `(30,)` | Source frame index of each step |
| `scene.ply` | | Start object (green), end object (yellow), first/last hand and predicted grippers (light to dark blue). Written with `--vis` |

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | | Dataset root |
| `--cases` | | Episodes, for example `1_16/977_9_10` |
| `--case_list` | | Text file with one episode per line |
| `--ckpt_dir` | `checkpoints` | Directory with `stage1.pt` and `stage2.pt` |
| `--out_dir` | `outputs` | Output directory |
| `--batch_size` | `8` | Episodes per batch |
| `--seed` | `0` | Sampling seed |
| `--vis` | off | Also write `scene.ply` |
| `--refine_endpoints` / `--no-refine_endpoints` | on | Translate the first and last grasp for contact alignment and to resolve gripper–object penetration. Orientation is unchanged |

## Evaluation

The test split `splits/test.txt` has 186 episodes over 40 objects. None of these objects appear in training.

```bash
for seed in 0 1 2 3; do
    python eval.py --data_root /path/to/CosmoH2G --seed $seed
done
python eval.py --summarize
```

Seeds can run in parallel on different GPUs. Per-episode results go to `results/seed<k>/metrics.jsonl`, and `--summarize` writes the mean over seeds to `results/summary.json`. Endpoint refinement is on by default here as well; pass `--no-refine_endpoints` to disable it.

| Metric | Description |
|--------|-------------|
| GOA (°) ↓ | Orientation error of the initial grasp |
| TS ↑ | Trajectory similarity of the fingertip midpoint |
| TOPA (°) ↓ | Orientation error of the terminal grasp |