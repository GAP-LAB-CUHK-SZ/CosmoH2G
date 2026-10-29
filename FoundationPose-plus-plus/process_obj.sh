#!/usr/bin/env bash
# Register the object pose on the start frame and the end frame of one case.
#
#   bash process_obj.sh \
#       --start-rgb /path/to/start.jpg \
#       --end-rgb /path/to/end.jpg \
#       --start-depth /path/to/start.raw \
#       --end-depth /path/to/end.raw \
#       --mesh /path/to/mesh.obj \
#       --start-mask /path/to/start_obj_mask.jpg \
#       --end-mask /path/to/end_obj_mask.jpg \
#       --output /path/to/output
#
# --cam-k is optional. A 3x3 text file is accepted; the default matches haptic/cam_K.txt.
set -euo pipefail

START_RGB=""
END_RGB=""
START_DEPTH=""
END_DEPTH=""
MESH=""
START_MASK=""
END_MASK=""
OUTPUT=""
CAM_K_FILE=""

usage() {
    sed -n '2,12p' "$0"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-rgb) START_RGB="${2:-}"; shift 2 ;;
        --end-rgb) END_RGB="${2:-}"; shift 2 ;;
        --start-depth) START_DEPTH="${2:-}"; shift 2 ;;
        --end-depth) END_DEPTH="${2:-}"; shift 2 ;;
        --mesh) MESH="${2:-}"; shift 2 ;;
        --start-mask) START_MASK="${2:-}"; shift 2 ;;
        --end-mask) END_MASK="${2:-}"; shift 2 ;;
        --output) OUTPUT="${2:-}"; shift 2 ;;
        --cam-k) CAM_K_FILE="${2:-}"; shift 2 ;;
        -h|--help) usage ;;
        *)
            echo "未知参数: $1"
            usage
            ;;
    esac
done

if [[ -z "$START_RGB" || -z "$END_RGB" || -z "$START_DEPTH" || -z "$END_DEPTH" || -z "$MESH" || -z "$START_MASK" || -z "$END_MASK" || -z "$OUTPUT" ]]; then
    echo "请传入 --start-rgb、--end-rgb、--start-depth、--end-depth、--mesh、--start-mask、--end-mask 和 --output"
    usage
fi

for path in "$START_RGB" "$END_RGB" "$START_DEPTH" "$END_DEPTH" "$MESH" "$START_MASK" "$END_MASK"; do
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

SCALE_FILE="$OUTPUT/obj_scale.txt"
START_POSE="$OUTPUT/start_obj_pose.npy"
END_POSE="$OUTPUT/end_obj_pose.npy"
START_VIS="$OUTPUT/start_obj_track"
END_VIS="$OUTPUT/end_obj_track"

python scripts/mesh_scale.py \
    --rgb_path "$START_RGB" \
    --raw_path "$START_DEPTH" \
    --obj_path "$MESH" \
    --mask_path "$START_MASK" \
    --cam_K "$CAM_K" \
    --output_file "$SCALE_FILE"
scale_value="$(cat "$SCALE_FILE")"
echo "scale_value: $scale_value"

python src/obj_pose_track.py \
    --rgb_seq_path "$START_RGB" \
    --depth_seq_path "$START_DEPTH" \
    --mesh_path "$MESH" \
    --init_mask_path "$START_MASK" \
    --pose_output_path "$START_POSE" \
    --mask_visualization_path "" \
    --bbox_visualization_path "" \
    --pose_visualization_path "$START_VIS" \
    --cam_K "$CAM_K" \
    --activate_2d_tracker \
    --activate_kalman_filter \
    --kf_measurement_noise_scale 0.05 \
    --apply_scale "$scale_value" \
    --force_apply_color \
    --apply_color "[0, 159, 237]" \
    --est_refine_iter 10 \
    --track_refine_iter 3

python src/obj_pose_track.py \
    --rgb_seq_path "$END_RGB" \
    --depth_seq_path "$END_DEPTH" \
    --mesh_path "$MESH" \
    --init_mask_path "$END_MASK" \
    --pose_output_path "$END_POSE" \
    --mask_visualization_path "" \
    --bbox_visualization_path "" \
    --pose_visualization_path "$END_VIS" \
    --cam_K "$CAM_K" \
    --activate_2d_tracker \
    --activate_kalman_filter \
    --kf_measurement_noise_scale 0.05 \
    --apply_scale "$scale_value" \
    --force_apply_color \
    --apply_color "[0, 159, 237]" \
    --est_refine_iter 10 \
    --track_refine_iter 3

echo "完成: $OUTPUT"
