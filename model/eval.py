"""Evaluate CosmoH2G on the test split and report GOA / TS / TOPA.

Usage:
    python eval.py --data_root /path/to/CosmoH2G --seed 0
    python eval.py --data_root /path/to/CosmoH2G --seed 1
    ...
    python eval.py --summarize            # average all finished seeds
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from cosmoh2g import CosmoH2G
from cosmoh2g.data import load_case, read_case_list, read_mesh
from cosmoh2g.metrics import grasp_orientation_accuracy, terminal_orientation_accuracy, trajectory_similarity

ROOT = Path(__file__).resolve().parent
METRICS = ("goa_deg", "ts", "topa_deg")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="Unpacked CosmoH2G release root")
    parser.add_argument("--ckpt_dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--split", type=Path, default=ROOT / "splits" / "test.txt")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--refine_endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Contact / penetration refinement of the first and last grasp",
    )
    parser.add_argument("--summarize", action="store_true", help="Only average the finished seeds")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    o3d.utility.random.seed(seed)


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(out_dir, n_cases):
    seeds = {}
    for path in sorted(out_dir.glob("seed*/metrics.jsonl")):
        rows = read_rows(path)
        if len(rows) == n_cases:
            seeds[path.parent.name] = {key: float(np.mean([row[key] for row in rows])) for key in METRICS}
    if not seeds:
        print("No finished seed.")
        return
    for name, values in seeds.items():
        print(f"{name}: GOA {values['goa_deg']:.2f} | TS {values['ts']:.4f} | TOPA {values['topa_deg']:.2f}")
    mean = {key: float(np.mean([values[key] for values in seeds.values()])) for key in METRICS}
    print(f"mean over {len(seeds)} seed(s): GOA {mean['goa_deg']:.2f} | TS {mean['ts']:.4f} | TOPA {mean['topa_deg']:.2f}")
    (out_dir / "summary.json").write_text(json.dumps({"n_cases": n_cases, "seeds": seeds, "mean": mean}, indent=2) + "\n")


def main():
    args = parse_args()
    cases = read_case_list(args.split)
    if args.summarize:
        summarize(args.out_dir, len(cases))
        return
    if args.data_root is None:
        raise SystemExit("--data_root is required")

    seed_dir = args.out_dir / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    result_path = seed_dir / "metrics.jsonl"
    done = {row["case"] for row in read_rows(result_path)}
    pending = [case for case in cases if case not in done]
    print(f"seed {args.seed}: {len(cases)} cases, {len(pending)} pending", flush=True)

    set_seed(args.seed)
    umi_verts, _ = read_mesh(os.path.join(args.data_root, "umi.obj"))
    model = CosmoH2G(args.ckpt_dir, umi_verts=umi_verts)
    for start in range(0, len(pending), args.batch_size):
        items = [load_case(args.data_root, case) for case in pending[start:start + args.batch_size]]
        preds = model.predict(items, refine=args.refine_endpoints)
        with result_path.open("a") as handle:
            for item, pred in zip(items, preds):
                gt = item["gt_keypoints"]
                row = {
                    "case": item["case"],
                    "goa_deg": grasp_orientation_accuracy(pred["start"], gt[0]),
                    "ts": trajectory_similarity(pred["keypoints"], gt),
                    "topa_deg": terminal_orientation_accuracy(pred["end"], gt[-1]),
                }
                handle.write(json.dumps(row) + "\n")
        print(f"seed {args.seed}: {min(start + args.batch_size, len(pending))}/{len(pending)}", flush=True)
    summarize(args.out_dir, len(cases))


if __name__ == "__main__":
    main()
