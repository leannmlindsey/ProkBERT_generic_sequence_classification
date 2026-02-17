#!/usr/bin/env python3
"""
Find the best seed from a set of fine-tuning runs by comparing test_results.json.

Usage:
    python find_best_seed.py /path/to/lr-1e-4_batch-32
    python find_best_seed.py /path/to/lr-1e-4_batch-32 --metric eval_f1
    python find_best_seed.py /path/to/lr-1e-4_batch-32 --metric eval_mcc --checkpoint_dir checkpoint-best
"""

import argparse
import json
import os
import sys
import glob


def main():
    parser = argparse.ArgumentParser(description="Find the best seed from fine-tuning runs")
    parser.add_argument("run_dir", help="Directory containing seed-* subdirectories")
    parser.add_argument("--metric", default="eval_mcc",
                        help="Metric to rank by (default: eval_mcc)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Checkpoint subdirectory name within each seed (e.g. checkpoint-best). "
                             "If not set, prints the seed directory itself.")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"ERROR: Directory not found: {args.run_dir}")
        sys.exit(1)

    # Find all seed directories with test_results.json
    seed_dirs = sorted(glob.glob(os.path.join(args.run_dir, "seed-*")),
                       key=lambda d: int(os.path.basename(d).split("-")[1]))

    if not seed_dirs:
        print(f"ERROR: No seed-* directories found in {args.run_dir}")
        sys.exit(1)

    results = []
    for seed_dir in seed_dirs:
        results_file = os.path.join(seed_dir, "test_results.json")
        if not os.path.isfile(results_file):
            print(f"  WARNING: No test_results.json in {seed_dir}, skipping")
            continue
        with open(results_file) as f:
            metrics = json.load(f)
        if args.metric not in metrics:
            print(f"  WARNING: Metric '{args.metric}' not found in {results_file}, skipping")
            continue
        results.append((seed_dir, metrics))

    if not results:
        print("ERROR: No valid test_results.json files found")
        sys.exit(1)

    # Sort by chosen metric descending
    results.sort(key=lambda x: x[1][args.metric], reverse=True)

    # Print summary table
    header_metrics = ["eval_accuracy", "eval_f1", "eval_mcc", "eval_auc", "eval_precision", "eval_recall"]
    # Ensure the ranking metric is shown
    if args.metric not in header_metrics:
        header_metrics.insert(0, args.metric)

    print(f"\n{'Seed':<10}", end="")
    for m in header_metrics:
        short = m.replace("eval_", "")
        print(f"{short:>12}", end="")
    print()
    print("-" * (10 + 12 * len(header_metrics)))

    for seed_dir, metrics in results:
        seed_name = os.path.basename(seed_dir)
        print(f"{seed_name:<10}", end="")
        for m in header_metrics:
            val = metrics.get(m, float("nan"))
            print(f"{val:>12.4f}", end="")
        print()

    # Best result
    best_dir, best_metrics = results[0]
    best_seed = os.path.basename(best_dir)

    print(f"\nBest seed by {args.metric}: {best_seed} ({best_metrics[args.metric]:.4f})")

    # Determine checkpoint path
    if args.checkpoint_dir:
        checkpoint_path = os.path.join(best_dir, args.checkpoint_dir)
    else:
        # Look for a checkpoint-* subdirectory automatically
        checkpoints = sorted(glob.glob(os.path.join(best_dir, "checkpoint-*")))
        if checkpoints:
            checkpoint_path = checkpoints[-1]  # latest checkpoint
        else:
            checkpoint_path = best_dir

    print(f"Checkpoint path: {checkpoint_path}")


if __name__ == "__main__":
    main()
