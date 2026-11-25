# CosmoH2G: Hand-to-Gripper Transfer for Complex Spatial Manipulation

<p align="center">
  <a href="https://cosmoh2g.github.io/"><img src="https://img.shields.io/badge/🌐-Website-blue"></a>
  <a href="https://arxiv.org/abs/2609.07498"><img src="https://img.shields.io/badge/📄-arXiv-red"></a>
</p>

<p align="center">
  <b>CosmoH2G</b> transfers complex spatial hand demonstrations to robotic grippers<br>
  through a two-stage, data-driven framework.
</p>

---

<p align="center">
  <img src="assets/dataset.jpg" height="180">
  <img src="assets/method.jpg" height="180">
</p>

<p align="center">
  <b>Left:</b> Dataset — 6,189 episodes · 1,254 objects &nbsp;|&nbsp;
  <b>Right:</b> Method — Stage-I keyframes → Stage-II full sequence
</p>

## Dataset

The release has 6,189 episodes across 1,254 objects, grouped by capture date. Object meshes, the UMI mesh, and the MANO models are shared at the repository root. Files are on [ModelScope `GAP-Lab/CosmoH2G`](https://www.modelscope.cn/datasets/GAP-Lab/CosmoH2G).

```text
CosmoH2G/
├── 3d_seq/
│   ├── 12_16.zip
│   ├── 12_17.zip
│   ├── ...
│   └── 4_1.zip
├── meshes.zip
├── umi.obj
├── MANO_RIGHT.npz
└── MANO_LEFT.npz
```

`3d_seq/` contains 37 date archives: `12_16`–`12_20`, `12_22`–`12_29`, `1_4`–`1_8`, `1_14`–`1_17`, `1_23`, `1_26`, `2_28`, `3_2`–`3_6`, `3_9`–`3_14`, and `4_1`. Each archive unpacks as:

```text
<date>/
└── <object_id>_<slot_start>_<slot_end>/
    ├── umi_pose_seq.npy
    ├── mano_right.npy      # or mano_left.npy
    └── obj_pose_ends.npy
```

`meshes.zip` unpacks to `meshes/<object_id:06d>.ply`. The object id is the first field of the case directory name. Object `977` is `meshes/000977.ply`.

The three sequences have the same length. Only the object start and end poses are stored.

| File | Shape | Content |
|------|-------|---------|
| `umi_pose_seq.npy` | `(N, 7)` | `x y z qw qx qy qz`, world pose of `umi_base` |
| `mano_right.npy` / `mano_left.npy` | `(N, 100)` | translation (3) + wrist 6D + 15 finger joints × 6D + scale. The filename selects the hand. `betas` are 0 |
| `obj_pose_ends.npy` | `(2, 8)` | row 0 is the start pose, row 1 is the end pose. Each row is `x y z qw qx qy qz scale` |

World object vertices are `R(q) @ (scale * ply) + xyz`. Quaternions are `qw qx qy qz`. A 6D rotation is the first two columns of the rotation matrix.

### Rendering

`render.py` renders one episode. Put it in the dataset root, next to `3d_seq/`, `meshes/`, `umi.obj`, and the two MANO models. It reads only files inside that directory. Unpack `meshes.zip` and the date archive before rendering. Dependencies are `numpy`, `scipy`, `open3d`, and `ffmpeg`.

```bash
python render.py 3d_seq/1_16/977_9_10
python render.py 3d_seq/1_16/977_9_10 -o /tmp/977_9_10.mp4
```

| Argument | Description |
|----------|-------------|
| `case` | Episode directory, for example `3d_seq/1_16/977_9_10` |
| `-o`, `--out` | Output mp4. Default is `<date>_<case>.mp4` in the current directory |

The start object is green and the end object is orange. Both stay in every frame. The hand and the UMI gripper move on every frame. Video size is 960×640 at 30 fps.

## Baseline Model

Inference code, pretrained weights and the evaluation script for the two-stage baseline are in [`model/`](model/README.md).

## Raw Video Processing

### Data Format

Each case directory contains:

```text
case/
├── start_obj_mask.jpg
├── end_obj_mask.jpg
├── gripper_base_mask.jpg
└── table_mask.jpg
```

`start_obj_mask.jpg` and `end_obj_mask.jpg` mark the object on the first and last frames. `gripper_base_mask.jpg` marks the gripper base. `table_mask.jpg` marks the table. Camera intrinsics shared by these captures are in `haptic/cam_K.txt`. Gripper meshes are `/mnt/CosmoH2G/umi.obj` and `/mnt/CosmoH2G/umi_base.obj`. Sequence paths and these files are passed in by the user.

### Hand Reconstructing and Tracking

We reconstruct the hand in every case with [HaPTIC](https://github.com/JudyYe/haptic), then align the reconstructed hand meshes into the 3D scene through depth ICP.

#### Environment

Run inside the `haptic` conda environment, from the `haptic/` directory.

```bash
conda create -n haptic python=3.10 -y
conda activate haptic
cd haptic
bash scripts/one_click.sh
```

`scripts/one_click.sh` installs dependencies and downloads the HaPTIC pretrained weights. Download the MANO model from the [MANO website](https://mano.is.tue.mpg.de/) and place it under `haptic/assets/mano/`.

#### Run

Pass the RGB directory, the depth directory, and the output directory.

```bash
cd haptic
conda activate haptic

python process_hand_worker.py \
    --rgb /path/to/color \
    --depth /path/to/depth \
    --output /path/to/output \
    --cam-k cam_K.txt
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--rgb` | | RGB frame directory |
| `--depth` | | Depth frame directory |
| `--output` | | Writes `haptic_out/` and `haptic_align_out/` here |
| `--cam-k` | `cam_K.txt` | Camera intrinsics |

### 2. Registering the Starting and Ending poses of Object

Run from `FoundationPose-plus-plus/`. Installation is described in [FoundationPose-plus-plus/Install.md](FoundationPose-plus-plus/Install.md). This step registers the starting frame and the ending frame, and estimates the object scale first.

```bash
cd FoundationPose-plus-plus

bash process_obj.sh \
    --start-rgb /path/to/start.jpg \
    --end-rgb /path/to/end.jpg \
    --start-depth /path/to/start.raw \
    --end-depth /path/to/end.raw \
    --mesh /path/to/mesh.obj \
    --start-mask /path/to/start_obj_mask.jpg \
    --end-mask /path/to/end_obj_mask.jpg \
    --output /path/to/output
```

Outputs are `obj_scale.txt`, `start_obj_pose.npy`, `end_obj_pose.npy`, `start_obj_track/`, and `end_obj_track/` under `--output`.

### 3. Tracking Gripper Motions

Tracks the gripper base over one RGB and depth sequence.

```bash
cd FoundationPose-plus-plus

bash process_gripper.sh \
    --rgb /path/to/color \
    --depth /path/to/depth \
    --mask /path/to/gripper_base_mask.jpg \
    --mesh /path/to/umi_base.obj \
    --output /path/to/output
```

Outputs are `base_pose.npy` and `base_track/` under `--output`.

## Citation

```bibtex
@misc{zhao2026cosmoh2ghandtogrippertransferdataset,
      title={CosmoH2G: A Hand-to-Gripper Transfer Dataset and Baseline Method for Object Manipulation with Complex Spatial Movements}, 
      author={Hongxiang Zhao and Mutian Xu and Zeyu Jin and Yiming Hao and Shuguang Cui and Xiaoguang Han},
      year={2026},
      eprint={2609.07498},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2609.07498}, 
}
