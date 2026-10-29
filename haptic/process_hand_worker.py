#!/usr/bin/env python
"""Hand reconstruction and depth alignment for one case.

Pass the RGB directory, the depth directory, and the output directory.
A JSONL manifest is still accepted so the date-batch scripts can reuse
this process. ViTDet / ViTPose / HaPTIC are loaded once. Detection is
batched and cached. Alignment stays in align_hand_mesh.py and runs only
after the resident models have been moved off the GPU.
"""

import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import time
import traceback

import torch
from omegaconf import OmegaConf

ROOT = osp.dirname(osp.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="单条 case 的手部重建与对齐")
    parser.add_argument("--rgb", help="RGB 帧目录")
    parser.add_argument("--depth", help="depth 帧目录")
    parser.add_argument("--output", help="输出目录，其下写入 haptic_out 与 haptic_align_out")
    parser.add_argument("--manifest", help="JSONL，每行一个 case。与 --rgb/--depth/--output 二选一")
    parser.add_argument("--cam-k", default=osp.join(ROOT, "cam_K.txt"))
    parser.add_argument("--hand-side", default="auto", choices=["both", "left", "right", "auto"])
    parser.add_argument("--det-batch", type=int, default=4)
    parser.add_argument("--run-viz", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=1)
    parser.add_argument("--end-idx", type=int, default=-2)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = parser.parse_args()
    single = any(value is not None for value in (args.rgb, args.depth, args.output))
    if args.manifest and single:
        parser.error("不要同时传入 --manifest 和 --rgb/--depth/--output")
    if not args.manifest and not (args.rgb and args.depth and args.output):
        parser.error("请传入 --rgb、--depth 和 --output")
    if not osp.isfile(args.cam_k):
        parser.error(f"找不到相机内参: {args.cam_k}")
    return args


def case_from_paths(args):
    rgb = osp.abspath(args.rgb)
    depth = osp.abspath(args.depth)
    output = osp.abspath(args.output)
    if not osp.isdir(rgb):
        raise SystemExit(f"RGB 目录不存在: {rgb}")
    if not osp.isdir(depth):
        raise SystemExit(f"depth 目录不存在: {depth}")
    name = osp.basename(output.rstrip(os.sep)) or "case"
    return {
        "folder": name,
        "rgb": rgb,
        "depth": depth,
        "save": osp.join(output, "haptic_out"),
        "align": osp.join(output, "haptic_align_out"),
        "img_root": osp.dirname(rgb.rstrip(os.sep)),
        "hand_side": args.hand_side,
        "start_idx": int(args.start_idx),
        "end_idx": int(args.end_idx),
    }


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_base_cfg():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_dir = osp.join(ROOT, "haptic", "configs_hydra")
    initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = compose(
        config_name="demo",
        overrides=["expname=release/mix_all", "cleanup=false"],
    )
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    return cfg


def case_cfg(base_cfg, case, det_batch):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    OmegaConf.set_struct(cfg, False)
    cfg.data.video_dir = case["rgb"]
    cfg.data.hand_side = case["hand_side"]
    cfg.data.start_idx = int(case["start_idx"])
    cfg.data.end_idx = int(case["end_idx"])
    cfg.save_dir = case["save"]
    cfg.cleanup = False
    cfg.det_cache_path = osp.join(case["save"], "det_cache.pkl")
    cfg.det_batch_size = int(det_batch)
    return cfg


def move_module(module, device):
    if module is None:
        return
    module.to(device)


def park_on_cpu(models):
    move_module(models["detector"].model, "cpu")
    move_module(models["cpm"].model, "cpu")
    models["cpm"].device = torch.device("cpu")
    move_module(models["haptic"], "cpu")
    move_module(models["wrapper"], "cpu")
    torch.cuda.empty_cache()


def stage_detector(models):
    move_module(models["haptic"], "cpu")
    move_module(models["wrapper"], "cpu")
    move_module(models["detector"].model, "cuda")
    models["detector"].model.eval()
    move_module(models["cpm"].model, "cuda")
    models["cpm"].device = torch.device("cuda")
    torch.cuda.empty_cache()


def stage_haptic(models):
    move_module(models["detector"].model, "cpu")
    move_module(models["cpm"].model, "cpu")
    models["cpm"].device = torch.device("cpu")
    move_module(models["haptic"], "cuda")
    models["haptic"].eval()
    move_module(models["wrapper"], "cuda")
    torch.cuda.empty_cache()


def align_done(case):
    summary = osp.join(case["align"], "align_summary.json")
    return osp.isfile(summary) and os.path.getsize(summary) > 0


def infer_ready(case):
    marker = osp.join(case["save"], "infer_done.json")
    if not osp.isfile(marker):
        return False
    try:
        with open(marker, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        meta.get("rgb") == case["rgb"]
        and int(meta.get("start_idx", -999)) == int(case["start_idx"])
        and int(meta.get("end_idx", -999)) == int(case["end_idx"])
        and meta.get("hand_side") == case["hand_side"]
    )


def write_infer_marker(case):
    marker = osp.join(case["save"], "infer_done.json")
    payload = {
        "rgb": case["rgb"],
        "start_idx": int(case["start_idx"]),
        "end_idx": int(case["end_idx"]),
        "hand_side": case["hand_side"],
    }
    tmp = marker + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp, marker)


def run_cmd(cmd):
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"命令失败 exit={proc.returncode}: {' '.join(cmd)}")


def run_align(case, cam_k, models):
    """在当前进程里对齐，复用已经加载的检测器。"""
    import align_hand_mesh

    stage_detector(models)
    args = align_hand_mesh.parse_args(
        [
            "--save_demo_dir",
            case["save"],
            "--all_hands",
            "--cam_K",
            cam_k,
            "--depth_dir",
            case["depth"],
            "--colors_dir",
            case["rgb"],
            "--output_dir",
            case["align"],
            "--mask_source",
            "image",
            "--export_obj",
        ]
    )
    align_hand_mesh.execute(args)


def run_viz(case):
    cmd = [
        sys.executable,
        osp.join(ROOT, "visualize_mesh_on_rgb.py"),
        "--save_demo_dir",
        case["save"],
        "--img_root",
        case["img_root"],
    ]
    if case["hand_side"] == "both":
        cmd.append("--bimanual")
    else:
        cmd.append("--all_seqs")
    run_cmd(cmd)


def process_case(case, models, base_cfg, det_batch, cam_k, run_viz_flag, gpu):
    name = case["folder"]
    if align_done(case):
        print(f"[gpu {gpu}] 跳过 {name}: 已有 align_summary.json", flush=True)
        return "skip"

    os.makedirs(case["save"], exist_ok=True)
    cfg = case_cfg(base_cfg, case, det_batch)
    started = time.time()

    if infer_ready(case):
        print(f"[gpu {gpu}] {name}: 复用已完成的 haptic_out", flush=True)
    else:
        from nnutils.det_utils import parse_det_seq
        from demo import infer_seq

        print(f"[gpu {gpu}] {name}: 检测 {case['rgb']}", flush=True)
        stage_detector(models)
        det_started = time.time()
        parse_det_seq(
            cfg.data,
            cfg,
            detector=models["detector"],
            cpm=models["cpm"],
            cache_path=str(cfg.det_cache_path),
            det_batch_size=det_batch,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[gpu {gpu}] {name}: 检测 {time.time() - det_started:.1f}s", flush=True)

        print(f"[gpu {gpu}] {name}: 手部推理", flush=True)
        stage_haptic(models)
        infer_started = time.time()
        infer_seq(cfg, model=models["haptic"], hand_wrapper=models["wrapper"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        write_infer_marker(case)
        print(f"[gpu {gpu}] {name}: 推理 {time.time() - infer_started:.1f}s", flush=True)

    print(f"[gpu {gpu}] {name}: 对齐", flush=True)
    park_on_cpu(models)
    align_started = time.time()
    run_align(case, cam_k, models)
    print(f"[gpu {gpu}] {name}: 对齐 {time.time() - align_started:.1f}s", flush=True)

    if run_viz_flag:
        print(f"[gpu {gpu}] {name}: 叠视频", flush=True)
        viz_started = time.time()
        run_viz(case)
        print(f"[gpu {gpu}] {name}: 叠视频 {time.time() - viz_started:.1f}s", flush=True)

    print(
        f"[gpu {gpu}] 完成 {name}: {time.time() - started:.1f}s",
        flush=True,
    )
    return "ok"


def load_models(base_cfg):
    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    from demo import load_haptic_model
    from nnutils.det_utils import get_cached_detector
    from nnutils.hand_utils import ManopthWrapper

    print("加载检测器...", flush=True)
    detector, cpm = get_cached_detector()
    print("加载 haptic 到 CPU...", flush=True)
    ckpt = base_cfg.ckpt
    if not osp.isabs(ckpt):
        ckpt = osp.join(ROOT, ckpt)
    haptic = load_haptic_model(ckpt, "cpu")
    wrapper = ManopthWrapper(base_cfg.paths.mano_dir).to("cpu")
    park_on_cpu(
        {
            "detector": detector,
            "cpm": cpm,
            "haptic": haptic,
            "wrapper": wrapper,
        }
    )
    return {
        "detector": detector,
        "cpm": cpm,
        "haptic": haptic,
        "wrapper": wrapper,
    }


def main():
    args = parse_args()
    cases = load_cases(args.manifest) if args.manifest else [case_from_paths(args)]
    print(
        f"[gpu {args.gpu}] cases={len(cases)} det_batch={args.det_batch} run_viz={args.run_viz}",
        flush=True,
    )
    if not cases:
        print(f"[gpu {args.gpu}] 没有待处理 case", flush=True)
        return 0

    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    base_cfg = build_base_cfg()
    models = load_models(base_cfg)

    ok, skipped, failed = [], [], []
    for case in cases:
        try:
            state = process_case(
                case,
                models,
                base_cfg,
                args.det_batch,
                args.cam_k,
                bool(args.run_viz),
                args.gpu,
            )
        except Exception:
            traceback.print_exc()
            failed.append(case["folder"])
            try:
                park_on_cpu(models)
            except Exception:
                traceback.print_exc()
            continue
        if state == "ok":
            ok.append(case["folder"])
        else:
            skipped.append(case["folder"])

    print(
        f"[gpu {args.gpu}] RESULT ok={len(ok)} skip={len(skipped)} fail={len(failed)}",
        flush=True,
    )
    if failed:
        print("[gpu {}] 失败: {}".format(args.gpu, " ".join(failed)), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
