#!/usr/bin/env bash
# Track the gripper base over one RGB/depth sequence.
#
#   bash process_gripper.sh \
#       --rgb /path/to/color \
#       --depth /path/to/depth \
#       --mask /path/to/gripper_base_mask.jpg \
#       --mesh /path/to/umi_base.obj \
#       --output /path/to/output
#
# --cam-k is optional. A 3x3 text file is accepted; the default matches haptic/cam_K.txt.
set -euo pipefail

RGB=""
DEPTH=""
MASK=""
MESH=""
OUTPUT=""
CAM_K_FILE=""

usage() {
    sed -n '2,10p' "$0"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rgb) RGB="${2:-}"; shift 2 ;;
        --depth) DEPTH="${2:-}"; shift 2 ;;
        --mask) MASK="${2:-}"; shift 2 ;;
        --mesh) MESH="${2:-}"; shift 2 ;;
        --output) OUTPUT="${2:-}"; shift 2 ;;
        --cam-k) CAM_K_FILE="${2:-}"; shift 2 ;;
        -h|--help) usage ;;
        *)
            echo "未知参数: $1"
            usage
            ;;
    esac
done

if [[ -z "$RGB" || -z "$DEPTH" || -z "$MASK" || -z "$MESH" || -z "$OUTPUT" ]]; then
    echo "请传入 --rgb、--depth、--mask、--mesh 和 --output"
    usage
fi

if [[ ! -d "$RGB" ]]; then
    echo "找不到 RGB 目录: $RGB"
    exit 1
fi
if [[ ! -d "$DEPTH" ]]; then
    echo "找不到 depth 目录: $DEPTH"
    exit 1
fi
for path in "$MASK" "$MESH"; do
    if [[ ! -f "$path" ]]; then
        echo "找不到文件: $path"
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
mkdir -p "$OUTPUT"

if [[ -n "$CAM_K_FILE" ]]; then
    if [[ ! -f "$CAM_K_FILE" ]]; then
        echo "找不到相机内参: $CAM_K_FILE"
        exit 1
    fi
    CAM_K="$(python -c 'import json,sys,numpy as np; print(json.dumps(np.loadtxt(sys.argv[1]).reshape(3, 3).tolist()))' "$CAM_K_FILE")"
else
    CAM_K="[[693.320984, 0, 645.644165],[0, 693.493652, 361.550293],[0, 0, 1]]"
fi

python src/obj_pose_track.py \
    --start_idx "1" \
    --end_idx "-1" \
    --rgb_seq_path "$RGB" \
    --depth_seq_path "$DEPTH" \
    --mesh_path "$MESH" \
    --init_mask_path "$MASK" \
    --pose_output_path "$OUTPUT/base_pose.npy" \
    --mask_visualization_path "" \
    --bbox_visualization_path "" \
    --pose_visualization_path "$OUTPUT/base_track" \
    --cam_K "$CAM_K" \
    --activate_2d_tracker \
    --activate_kalman_filter \
    --kf_measurement_noise_scale 0.05 \
    --apply_scale 1.0 \
    --force_apply_color \
    --apply_color "[0, 159, 237]" \
    --est_refine_iter 10 \
    --track_refine_iter 3

echo "完成: $OUTPUT"
